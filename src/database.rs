use crate::hash::*;
use crate::clean::Cleaner;
use rusqlite::{self, params, Connection, OptionalExtension};
use tokio::sync::Mutex;
use anyhow::Result;

// Vector Strore and Embedding
use turbovec::TurboQuantIndex;
use fastembed::{TextEmbedding};

const CANDIDATE_POOL: usize = 20;
const NUMBER_OF_WORDS_PER_CHUNK: usize = 200;
const DOCUMENT_DEDUPLICATION: &str = "
    CREATE TABLE IF NOT EXISTS document_deduplication (
        hash INT PRIMARY KEY
    );
";

const DOCUMENT: &str = "
    CREATE VIRTUAL TABLE IF NOT EXISTS documents
    USING fts5 (text);
";

const INSERT: &str = "
    INSERT INTO documents (rowid, text)
    VALUES (?1, ?2);
";

const INSERT_DUPE: &str = "
    INSERT INTO document_deduplication (hash)
    VALUES (?1);
";

// Lexical (BM25) candidates: keyword matches ranked by relevance.
macro_rules! SELECT_LEXICAL { () => {"
    SELECT rowid, text, BM25(documents) AS rank
    FROM documents
    WHERE text MATCH ?1
    ORDER BY rank ASC
    LIMIT ?2;
"}; }

// Fetch the text for an explicit set of rowids (the semantic candidates).
macro_rules! SELECT_BY_ROWID { () => {"
    SELECT rowid, text
    FROM documents
    WHERE rowid IN ({});
"}; }

pub struct Database {
    connection: Mutex<Connection>,
    vector_store: Mutex<TurboQuantIndex>,
    embedding: Mutex<TextEmbedding>,
    cleaner: Cleaner,
}

trait JoinVeci64 {
    fn join(&self, delimiter: &str) -> String;
}

impl JoinVeci64 for Vec<i64> {
    fn join(&self, delimiter: &str) -> String {
        let strings: Vec<String> = self
            .iter()
            .map(|n| n.to_string())
            .collect();

        strings.join(delimiter)
    }
}

impl Database {
    pub fn new() -> Self {
        let db = Connection::open_in_memory().expect("database connection");
        let vector_store = TurboQuantIndex::new(384, 4).expect("vector store init");
        let embedding = TextEmbedding::try_new(Default::default()).expect("embedding model init");
        let cleaner = Cleaner::new();

        let _ = db.execute(DOCUMENT, ());
        let _ = db.execute(DOCUMENT_DEDUPLICATION, ());

        Self {
            connection: db.into(),
            vector_store: vector_store.into(),
            embedding: embedding.into(),
            cleaner,
        }
    }

    async fn check_duplicate(&self, text: &str) -> Result<bool> {
        let guard = self.connection.lock().await;
        let text_hash: i64 = hash(text);
	let duplicate: Option<String> = guard.query_row(
            "SELECT hash FROM document_deduplication WHERE hash = ?1 LIMIT 1;",
            params![text_hash],
            |row| row.get(0),
        ).optional()?;

        Ok(duplicate.is_some())
    }

    fn chunk(&self, text: &str) -> Vec<String> {
        let mut chunks: Vec<String> = vec![];
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_string())
            .collect();
        let number_of_words: usize = words.len();
        let number_of_chunks: usize = number_of_words / NUMBER_OF_WORDS_PER_CHUNK;

        for chunk in 0..number_of_chunks {
            let sentence: String = words[
                chunk*NUMBER_OF_WORDS_PER_CHUNK..
                (chunk+1)*NUMBER_OF_WORDS_PER_CHUNK
            ].join(" ");
            chunks.push(sentence);
        }
        let tail: &[String] = &words[number_of_chunks*NUMBER_OF_WORDS_PER_CHUNK..number_of_words];
        if !tail.is_empty() {
            chunks.push(tail.join(" "));
        }

        chunks
    }

    pub async fn add_document(&self, document: &str) -> Result<usize> {
        let chunks: Vec<String> = self.chunk(document);
        let mut inserted: usize = 0;
        for chunk in chunks {
            let cleaned: String = self.cleaner.clean(&chunk);
            self.insert(&cleaned).await?;
            inserted += 1;
        }
        Ok(inserted)
    }

    pub async fn insert(&self, text: &str) -> Result<i64> {
        if self.check_duplicate(text).await? {
            return Ok(0);
        }

        let input: Vec<&str> = vec![text];
        let mut embedding_guard = self.embedding.lock().await;
        let mut vector_guard = self.vector_store.lock().await;
        let guard = self.connection.lock().await;
        let vectors = embedding_guard.embed(input, None)?;
        let rowid: i64 = vector_guard.len() as i64 + 1;
        guard.execute(INSERT, params![rowid, text])?;
        let text_hash: i64 = hash(text);
        guard.execute(INSERT_DUPE, params![text_hash])?;

        vector_guard.add(&vectors[0]);

        Ok(rowid)
    }

    fn match_query(cleaned: &str) -> String {
        cleaned
            .split_whitespace()
            .map(|token| format!("\"{}\"", token.replace('"', "\"\"")))
            .collect::<Vec<String>>()
            .join(" OR ")
    }

    async fn semantic_candidates(&self, cleaned: &str) -> Result<Vec<i64>> {
        let input: Vec<&str> = vec![cleaned];
        let mut embedding_guard = self.embedding.lock().await;
        let vector_guard = self.vector_store.lock().await;
        let vectors = embedding_guard.embed(input, None)?;
        let results = vector_guard.search(&vectors[0], CANDIDATE_POOL);

        let rowids: Vec<i64> = results.indices.iter().map(|slot| slot + 1).collect();
        Ok(rowids)
    }

    async fn lexical_candidates(&self, cleaned: &str) -> Result<Vec<i64>> {
        let match_query: String = Self::match_query(cleaned);
        if match_query.is_empty() {
            return Ok(vec![]);
        }

        let guard = self.connection.lock().await;
        let mut statement = guard.prepare(SELECT_LEXICAL!())?;
        let rows = statement.query_map(
            params![match_query, CANDIDATE_POOL as i64],
            |row| row.get::<_, i64>(0),
        )?;

        let mut rowids: Vec<i64> = vec![];
        for row in rows {
            rowids.push(row?);
        }
        Ok(rowids)
    }

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

    pub async fn search(&self, search: &str, limit: i32) -> Result<String> {
        let cleaned: String = self.cleaner.clean(&search);
        let semantic: Vec<i64> = self.semantic_candidates(&cleaned).await?;
        let lexical: Vec<i64> = self.lexical_candidates(&cleaned).await?;
        let fused: Vec<i64> = Self::fuse(&[semantic, lexical]);

        let top: Vec<i64> = fused.into_iter().take(limit.max(0) as usize).collect();
        if top.is_empty() {
            return Ok(String::new());
        }

        let select = format!(SELECT_BY_ROWID!(), top.join(","));
        let guard = self.connection.lock().await;
        let mut statement = guard.prepare(&select)?;
        let rows = statement.query_map(params![], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut text_by_rowid: std::collections::HashMap<i64, String> =
            std::collections::HashMap::new();
        for row in rows {
            let (rowid, text) = row?;
            text_by_rowid.insert(rowid, text);
        }

        let docs: Vec<String> = top
            .iter()
            .filter_map(|rowid| text_by_rowid.remove(rowid))
            .collect();

        Ok(docs.join("\n"))
    }
}