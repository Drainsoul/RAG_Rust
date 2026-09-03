# RAG Pipeline Review — Rust + Gemma 4 E4B

A concept-by-concept breakdown of the RAG pipeline, cited to the exact file and line it lives in, plus reviewer notes on each piece.

## What is RAG?

**Retrieval-Augmented Generation** solves a basic problem: an LLM like Gemma only "knows" what was baked into its weights at training time. It doesn't know your private documents, and it can hallucinate facts it was never trained on. RAG fixes this by **retrieving relevant text from an external store first, then stuffing that text into the prompt** as grounding context before the model generates a response.

The pipeline follows the classic three-stage shape:

1. **Ingest** — chunk and index documents (`add_document`, `src/database.rs:126`)
2. **Retrieve** — given a query, find the most relevant chunks (`search`, `src/database.rs:213`)
3. **Generate** — hand those chunks + the question to Gemma (`ask`, `src/main.rs:57`, forwarding to `gemma/main.py:9`)

The `root` handler in `src/main.rs:76-88` is the glue: it takes a user prompt, calls `database.search()` to get the top 2 relevant chunks, concatenates them with the prompt, and sends the whole thing to the Gemma server.

---

## 1. Tokenization

**What it does:** breaks text into normalized units (tokens/words) so search and embedding models compare like-for-like — removing case differences, filler words ("the", "a", "is"), and grammatical suffixes so "running", "runs", and "run" all collapse to the same signal.

**Where it lives:** `src/clean.rs:24-35`, the `Cleaner::clean` method:

```rust
pub fn clean(&self, text: &str) -> String {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|word| word.to_lowercase())
        .filter(|word| !self.stop_words.contains(word))
        .map(|word| self.stemmer.stem(&word).to_string())
        .collect();
    words.join(" ")
}
```

Walking through it on real corpus data — take this line from `data/text.txt`:

> `"By 2050, AI architects will have designed self-constructing buildings."`

- `split_whitespace()` + `to_lowercase()` → `["by", "2050,", "ai", "architects", "will", "have", "designed", ...]`
- stop-word filter drops `by`, `will`, `have`
- the Porter stemmer (`Stemmer::create(Algorithm::English)`, `clean.rs:12`) strips `designed` → `design`, `architects` → `architect`

Result: `"2050, ai architect design self-construct build."`

This cleaned string is what actually gets embedded and indexed — not the raw sentence. See `database.rs:130` (`add_document`) and `database.rs:214` (`search`), both call `self.cleaner.clean(...)` before anything else happens.

**Reviewer note:** the pipeline cleans *both* the stored text before insertion (line 130) and the query before search (line 214) — good, that's necessary for BM25/embedding consistency. One catch: because it cleans before storing, the **original raw text is lost** — what gets returned to the user/LLM is the stemmed, stop-word-stripped version (`"2050, ai architect design..."`), not natural English. For a RAG context window that's usually undesirable — Gemma will receive garbled grammar as its grounding context. A more standard pattern is to store both: raw text for retrieval-and-display, and a separately cleaned copy just for indexing keys. Example:

```rust
pub async fn add_document(&self, document: &str) -> Result<usize> {
    let chunks: Vec<String> = self.chunk(document);
    let mut inserted: usize = 0;
    for chunk in chunks {
        let cleaned: String = self.cleaner.clean(&chunk);
        self.insert(&chunk, &cleaned).await?; // store raw + indexed-by-cleaned
        inserted += 1;
    }
    Ok(inserted)
}
```

---

## 2. Embedding

**What it does:** converts text into a fixed-length numeric vector (here, 384 dimensions) that captures *meaning* rather than exact wording, so "car" and "automobile" land near each other in vector space even though they share no characters.

**Where it lives:** `fastembed::TextEmbedding` (imported `database.rs:9`), instantiated with defaults at `database.rs:76`:

```rust
let embedding = TextEmbedding::try_new(Default::default()).expect("embedding model init");
```

Used in two places:
- **at insert time** — `database.rs:146`: `embedding_guard.embed(input, None)?`
- **at query time** — `database.rs:169`, inside `semantic_candidates`

Both produce a `Vec<f32>` per string, which then feeds the vector store (`vector_guard.add(&vectors[0])`, line 152).

**Reviewer note:** `TextEmbedding::try_new(Default::default())` silently picks whatever fastembed's default model is (recent fastembed-rs typically defaults to a small BGE or AllMiniLM variant producing 384-dim vectors — consistent with `TurboQuantIndex::new(384, 4)` at line 75, so the dimensions do line up correctly). If the embedding model is ever swapped, that constant `384` at line 75 has to change in lockstep or `vector_guard.add` will panic/corrupt the index. Worth pulling into a shared constant:

```rust
const EMBEDDING_DIM: usize = 384;
let vector_store = TurboQuantIndex::new(EMBEDDING_DIM, 4).expect("vector store init");
```

---

## 3. BM25 (lexical / keyword search)

**What it does:** a statistical ranking function over exact term matches — it scores a document higher when a query term appears frequently in that document but rarely across the whole corpus (TF-IDF, essentially, with length normalization). Good at catching exact keyword hits that embeddings can blur past (e.g., a specific product name or acronym).

**Where it lives:** delegated to **SQLite's FTS5 virtual table**, which has BM25 built in. Table setup: `database.rs:19-22`

```rust
const DOCUMENT: &str = "
    CREATE VIRTUAL TABLE IF NOT EXISTS documents
    USING fts5 (text);
";
```

The scoring query: `database.rs:35-41`

```rust
macro_rules! SELECT_LEXICAL { () => {"
    SELECT rowid, text, BM25(documents) AS rank
    FROM documents
    WHERE text MATCH ?1
    ORDER BY rank ASC
    LIMIT ?2;
"}; }
```

Called from `lexical_candidates`, `database.rs:176-194`. Note the `?1` param is built by `match_query`, `database.rs:157-163`:

```rust
fn match_query(cleaned: &str) -> String {
    cleaned
        .split_whitespace()
        .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
        .collect::<Vec<String>>()
        .join(" OR ")
}
```

So the cleaned query `"2050 ai architect design self-construct build"` becomes the FTS5 match expression `"2050" OR "ai" OR "architect" OR "design" OR ...` — an OR-of-terms search, letting any single keyword match pull a candidate in.

**Reviewer note:** `ORDER BY rank ASC` is correct — FTS5's `bm25()` returns *lower is better* (it's a cost/distance convention, not a similarity score), unlike most BM25 implementations elsewhere where higher = better. Easy place to introduce a silent bug if someone "fixes" this later without knowing FTS5's convention, so a one-line comment there would save a future debugging session.

---

## 4. Semantic Search

**What it does:** instead of matching exact words, it finds chunks whose *embedding vectors* are closest (by distance/cosine similarity) to the query's embedding — catching paraphrases and related concepts that share no vocabulary with the query.

**Where it lives:** `semantic_candidates`, `database.rs:165-174`

```rust
async fn semantic_candidates(&self, cleaned: &str) -> Result<Vec<i64>> {
    let input: Vec<&str> = vec![cleaned];
    let mut embedding_guard = self.embedding.lock().await;
    let vector_guard = self.vector_store.lock().await;
    let vectors = embedding_guard.embed(input, None)?;
    let results = vector_guard.search(&vectors[0], CANDIDATE_POOL);

    let rowids: Vec<i64> = results.indices.iter().map(|slot| slot + 1).collect();
    Ok(rowids)
}
```

Concretely: a query like `"future construction robots"` never appears verbatim in the corpus, but it embeds close to `"By 2050, AI architects will have designed self-constructing buildings."` because the model has learned that "construction," "robots," and "self-constructing buildings" occupy nearby regions of the embedding space. BM25 alone would likely miss this pairing entirely (zero shared keyword tokens after stemming); semantic search catches it.

The `slot + 1` on line 172 exists because the vector store is 0-indexed internally but SQLite `rowid`s in `insert()` are assigned starting at 1 (`let rowid: i64 = vector_guard.len() as i64 + 1;`, line 147) — so this off-by-one correction keeps the two indices aligned. Worth a comment there too, since it's a classic source of off-by-one bugs if either indexing scheme changes independently.

---

## 5. Turbovec Quantization

**What it does:** "quantization" here means compressing each 384-dimensional `f32` vector into a smaller representation (fewer bits per dimension) so the index fits in memory and searches faster, at the cost of a small amount of precision — a standard trade in vector databases (see also: Faiss's `IVF-PQ`, HNSW with scalar quantization, etc.).

**Where it lives:** `database.rs:8` (import), `database.rs:75` (construction):

```rust
use turbovec::TurboQuantIndex;
...
let vector_store = TurboQuantIndex::new(384, 4).expect("vector store init");
```

The `4` here is turbovec's quantization parameter (bits or subvector count, depending on the library's API — turbovec is a newer/niche crate so it's worth double-checking its docs for exactly what that second argument controls, since `Cargo.toml:19` pins it to `"*"`, meaning it'll silently pick up whatever the latest published version does with that parameter). Retrieval uses the quantized index directly: `vector_guard.search(&vectors[0], CANDIDATE_POOL)` at `database.rs:170`.

**Reviewer note:** `turbovec = "*"` and `fastembed = "*"` (`Cargo.toml:19-20`) are both unpinned. That's risky for exactly this component — quantization parameters, index formats, and default embedding models are the kind of thing that silently change between minor versions and can invalidate an existing index or shift the dimension count. Recommend pinning these:

```toml
turbovec = "0.x.y"   # pin to whatever is actually running
fastembed = "3.x.y"
```

---

## 6. Reciprocal Rank Fusion (RRF)

**What it does:** combines two *independently ranked* result lists (the BM25 list and the semantic list) into one merged ranking — without needing their scores to be on comparable scales (BM25 scores and cosine distances aren't directly comparable numbers). Each list contributes `1 / (k + rank)` per item it contains; items appearing near the top of *either* list, or in *both*, bubble to the top of the fused list.

**Where it lives:** `fuse`, `database.rs:196-211`

```rust
fn fuse(ranked_lists: &[Vec<i64>]) -> Vec<i64> {
    let mut scores: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
    for list in ranked_lists {
        for (rank, rowid) in list.iter().enumerate() {
            *scores.entry(*rowid).or_insert(0.0) += 1.0 / (60.0 + rank as f32);
        }
    }

    let mut fused: Vec<(i64, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    fused.into_iter().map(|(rowid, _)| rowid).collect()
}
```

`60.0` is the standard RRF constant `k` from the original RRF paper (Cormack et al.) — it dampens the effect of any single rank position so lower-ranked items still contribute a little, preventing the #1 spot in one list from completely dominating.

Called from `search`, `database.rs:213-217`:

```rust
let semantic: Vec<i64> = self.semantic_candidates(&cleaned).await?;
let lexical: Vec<i64> = self.lexical_candidates(&cleaned).await?;
let fused: Vec<i64> = Self::fuse(&[semantic, lexical]);
```

Worked example: say for query `"carbon offset"`:
- BM25 ranks `"Carbon offset programs will encourage..."` at rank 0 (exact keyword hit) → score `1/60 ≈ 0.01667`
- Semantic search ranks the same chunk at rank 2 (embeddings put a few paraphrases ahead of it) → score `1/62 ≈ 0.01613`
- Fused score: `0.01667 + 0.01613 ≈ 0.03280` — appearing in *both* lists nearly doubles its score, likely pushing it to the very top of the final ranking. That's exactly the hybrid-search behavior you want: a chunk that's both keyword-relevant and semantically relevant should outrank one that's only strong in a single signal.

This is a solid, textbook implementation of hybrid search — the same fusion strategy used in production systems like Elasticsearch's hybrid retrieval and Weaviate's hybrid search.

---

## 7. Feeding Gemma 4 E4B

Once `search()` returns the fused top-N chunks (`database.rs:213-244`, joined with `\n` at line 243), `main.rs` wires it into the prompt:

`src/main.rs:76-88`:

```rust
async fn root(
    state: web::Data<Arc<RAGState>>,
    body: web::Json<Value>,
) -> HttpResponse {
    let prompt: &str = body["prompt"].as_str().unwrap_or("");
    let docs = state.database.search(prompt, 2).await.unwrap_or("".to_string());
    let answer = ask(&format!("{docs}\n{prompt}")).await.unwrap_or("".to_string());
    ...
}
```

`ask()`, `main.rs:57-66`, ships that combined string over HTTP to the Python model server:

```rust
async fn ask(question: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let response: String = client
        .get(&format!("http://0.0.0.0:8000/{question}"))
        .send()
        .await?
        .text()
        .await?;
    Ok(response)
}
```

And `gemma/main.py:9-27` receives it as a path parameter, wraps it in a chat template, and generates:

```python
@app.get("/{prompt}")
async def root(prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(target_model.device)
    ...
```

So the full RAG loop is: **raw query → clean/tokenize → BM25 candidates + semantic candidates → RRF fusion → top-2 chunks → concatenated with query → sent as the `{prompt}` path segment → Gemma 4 E4B generates grounded answer.**

**Reviewer note — fix this one first:** passing `docs + "\n" + prompt` as a **URL path segment** (`main.rs:59`, `gemma/main.py:9` route `@app.get("/{prompt}")`) is fragile for RAG specifically, because retrieved chunks can easily contain characters that break URL parsing (`&`, `#`, `?`, newlines, non-ASCII) and most servers cap URL/path length well below what two 200-word chunks plus a question will produce. This will likely break or truncate silently on real documents. The fix is to send it as a POST body instead, matching the pattern already used for `/doc` and `/`:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AskRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def root(req: AskRequest):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": req.prompt},
    ]
    ...
```

```rust
async fn ask(question: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let response: String = client
        .post("http://0.0.0.0:8000/ask")
        .json(&serde_json::json!({ "prompt": question }))
        .send()
        .await?
        .text()
        .await?;
    Ok(response)
}
```

That one change removes a class of bugs that would otherwise show up as mysterious 404s or truncated context once the corpus grows past toy sentences.