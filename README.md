readme_content = """# Core Retrieval-Augmented Generation (RAG) Architecture & Indexing Components

This document provides a comprehensive technical guide to the foundational building blocks handling **data indexing, retrieval, vector compression, and hybrid scoring** in modern Retrieval-Augmented Generation (RAG) systems.

---

## Technical Overview

A robust RAG pipeline requires balancing **exact lexical matching**, **conceptual understanding**, **memory efficiency**, and **result fusion**. The following six components form the core system architecture:

1. **Tokenization & Embedding**: Structural Ingestion & Mapping
2. **BM25 Search**: Lexical / Keyword Retrieval
3. **Semantic Search**: High-Dimensional Conceptual Retrieval
4. **Turbovec Quantization**: Vector Index Memory Optimization
5. **Reciprocal Rank Fusion (RRF)**: Multi-Index Hybrid Re-Ranking
6. **End-to-End Execution Flow**: Pipeline Integration

---

## 1. Tokenization & Embedding: Structural Ingestion

### Tokenization
* **Definition**: The process of breaking continuous text strings into discrete algorithmic units (tokens) such as sub-words, words, or characters using algorithms like Byte-Pair Encoding (BPE), WordPiece, or SentencePiece.
* **Role in RAG**: Large Language Models (LLMs) and vector encoders cannot process raw strings directly. Tokenization defines chunk boundaries (e.g., 512-token sliding windows) and ensures text aligns with strict context window limits.
* **Concrete Example**:
  * **Raw Input**: `"unbelievable error code"`
  * **Sub-word Tokens**: `["un", "believ", "able", " error", " code"]`
  * **Vocabulary ID Mapping**: `[412, 8931, 245, 1043, 882]`

### Embedding
* **Definition**: Converts tokenized chunks into dense, fixed-size floating-point vectors (e.g., 768 or 1536 dimensions) using dedicated transformer-based embedding models (e.g., `text-embedding-3-large`, `bge-large-en-v1.5`).
* **Role in RAG**: Maps semantic meaning into a continuous vector space where semantically similar concepts sit in close proximity, enabling non-exact conceptual lookup.
* **Concrete Example**:
  * **Input Chunk**: `"User failed to authenticate"`
  * **Dense Vector Representation**: `[0.023, -0.412, 0.891, ..., 0.104]` ($\in \mathbb{R}^{1536}$)

---

## 2. BM25 Search: Keyword Retrieval

**BM25 (Best Matching 25)** is a sparse lexical retrieval algorithm based on the probabilistic relevance framework. It scores documents relative to a query based on term frequency (TF), inverse document frequency (IDF), term frequency saturation, and document length normalization.

$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

* **Purpose**: Dense embedding vectors frequently struggle with exact alphanumeric identifiers, serial numbers, code snippets, or rare domain-specific jargon. BM25 guarantees that exact term matches are strictly preserved.
* **Role in RAG**: Acts as the sparse retrieval engine in hybrid search pipelines, guaranteeing top-k precision for specific entities.
* **Concrete Example**:
  * **Query**: `"ERR-404-AUTH"`
  * **Behavior**: BM25 directly indexes the exact character string `"ERR-404-AUTH"` and prioritizes documents containing exact matches, whereas an embedding model might blur the term into generic "network authentication errors."

---

## 3. Semantic Search: Conceptual Retrieval

Semantic search measures the mathematical similarity (such as **Cosine Similarity** or **Inner Product**) between a query vector and indexed document vectors in high-dimensional vector space.

$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

* **Purpose**: Bypasses the limitations of keyword matching by capturing intent, context, and structural synonyms regardless of exact vocabulary overlap.
* **Role in RAG**: Retrieves document chunks based on conceptual relevance, matching user intent to knowledge base articles phrased completely differently.
* **Concrete Example**:
  * **Query**: `"My laptop screen is pitch black"`
  * **Retrieved Chunk**: `"Troubleshooting display power issues"`
  * **Result**: High cosine similarity score ($\approx 0.89$) despite sharing **zero** matching keywords.

---

## 4. Turbovec Quantization: Memory Optimization

**Turbovec Quantization** is a high-performance vector compression method based on the **TurboQuant** algorithm. It compresses high-dimensional `float32` vectors down to low-bit representations (e.g., 2-bit, 3-bit, or 4-bit) without requiring model fine-tuning or retraining.

* **Mechanism**: Applies a random orthogonal rotation to project vector dimensions into a uniform, predictable coordinate distribution, followed by data-oblivious Lloyd-Max scalar quantization.
* **Role in RAG**: Solves the severe RAM bottlenecks associated with large vector databases, reducing memory consumption by up to **16x** with negligible accuracy loss ($< 5\%$).
* **Concrete Example & Memory Comparison**:
  * **Dataset**: $10,000,000$ document vectors ($1536$ dimensions each).
  * **Uncompressed (`FP32`)**: $10,000,000 \times 1536 \times 4 \text{ bytes} \approx \mathbf{61.44 \text{ GB RAM}}$
  * **Turbovec Quantized (3-bit)**: Compresses vector representation down to $\approx \mathbf{5.7 \text{ GB RAM}}$ (Fits on standard low-cost instances while maintaining $95\%+$ retrieval precision).

---

## 5. Reciprocal Rank Fusion (RRF): Hybrid Scoring

**Reciprocal Rank Fusion (RRF)** is a robust, model-agnostic rank aggregation algorithm that merges candidate lists from multiple retrieval systems (e.g., BM25 and Semantic Search) without requiring score normalization.

### RRF Formula

$$\text{RRF\_Score}(d \in D) = \sum_{m \in R} \frac{1}{k + r_m(d)}$$

Where:
* $R$: The set of retrieval systems (e.g., $\{\text{BM25}, \text{Semantic}\}$).
* $r_m(d)$: The 1-based rank of document $d$ in retrieval system $m$.
* $k$: A smoothing constant (standard benchmark default: $k = 60$).

---

### Step-by-Step RRF Calculation Example

Consider two documents evaluated across **BM25** (Sparse) and **Semantic Search** (Dense):

#### Candidate Evaluation Table

| Document | Rank in BM25 ($r_1$) | Rank in Semantic ($r_2$) |
| :--- | :---: | :---: |
| **Document A** | 1 | 10 |
| **Document B** | 2 | 2 |

#### Calculation Details ($k = 60$)

1. **Document A**:
   $$\text{RRF}(A) = \frac{1}{60 + 1} + \frac{1}{60 + 10} = \frac{1}{61} + \frac{1}{70} \approx 0.01639 + 0.01428 = \mathbf{0.03067}$$

2. **Document B**:
   $$\text{RRF}(B) = \frac{1}{60 + 2} + \frac{1}{60 + 2} = \frac{1}{62} + \frac{1}{62} = \frac{2}{62} \approx \mathbf{0.03225}$$

#### Final Result
**Document B** wins ($\mathbf{0.03225} > \mathbf{0.03067}$) because it achieved consistent, high placement across both lexical and conceptual retrieval streams.

---

## 6. How They Work Together: End-to-End RAG Pipeline