# Cross-Lingual Academic Information Retrieval

A multilingual academic information retrieval system that allows users
to search an academic passage collection in **English, Hindi, Bengali,
or Telugu**, even when the query language and corpus language are
different.

The project compares traditional lexical retrieval with multilingual
semantic retrieval and adds translation-assisted search and rank fusion
to improve cross-lingual retrieval.

![Project UI](public/image.png)

------------------------------------------------------------------------

## Overview

Conventional keyword search works well when the query and document use
the same vocabulary. It becomes much less effective when the query is
written in one language and the relevant document is written in another.

This project addresses that problem using a combination of:

-   **BM25** for lexical retrieval
-   **Multilingual E5 embeddings** for semantic cross-lingual retrieval
-   **NLLB-200** for query translation
-   **Reciprocal Rank Fusion (RRF)** for combining retrieval signals
-   Optional **BGE cross-encoder reranking**
-   A **FastAPI backend** and **React + Vite frontend**

The system supports all **16 query-language → corpus-language
combinations** among English, Hindi, Bengali, and Telugu. The main
research focus is the **12 cross-lingual combinations**.

------------------------------------------------------------------------

## Key Features

-   🔎 Search across English, Hindi, Bengali, and Telugu
-   🌐 Cross-lingual retrieval: query and corpus can use different
    languages
-   🧠 Multilingual semantic search with E5 embeddings
-   🔤 BM25 lexical baseline for comparison
-   🔄 NLLB-200 query translation for cross-lingual retrieval
-   ⚡ In-memory translation caching
-   🧩 Weighted Reciprocal Rank Fusion
-   🎯 Optional cross-encoder reranking
-   🗂️ Passage details and language-aware results
-   📊 Evaluation using Recall@1, Recall@10, and MRR@10
-   🧪 Evaluation across all 16 language-pair combinations
-   📈 Saved experimental results and methodology documentation
-   🖥️ Web interface built with React and Vite
-   🚀 REST API built with FastAPI

------------------------------------------------------------------------

## System Architecture

``` text
                         ┌─────────────────────┐
                         │       User Query    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Language Detection │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          Same-language search              Cross-lingual search
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  NLLB-200        │
                                          │ Query Translation│
                                          └────────┬─────────┘
                                                   │
                         ┌─────────────────────────┼────────────────────────┐
                         │                         │                        │
                         ▼                         ▼                        ▼
                  Dense Retrieval          Translated Dense          Translated BM25
                  Original Query              Retrieval
                         │                         │                        │
                         └─────────────────────────┼────────────────────────┘
                                                   ▼
                                      ┌─────────────────────────┐
                                      │ Reciprocal Rank Fusion  │
                                      └────────────┬────────────┘
                                                   │
                                                   ▼
                                      ┌─────────────────────────┐
                                      │ Optional Cross-Encoder   │
                                      │      Re-ranking          │
                                      └────────────┬────────────┘
                                                   │
                                                   ▼
                                      ┌─────────────────────────┐
                                      │       Top-K Results      │
                                      └─────────────────────────┘
```

------------------------------------------------------------------------

## Retrieval Modes

The API exposes three retrieval modes.

### 1. BM25

Traditional lexical retrieval using Okapi BM25.

Useful as a baseline and effective for same-language queries, but
cross-lingual performance is limited because different scripts have
little lexical overlap.

### 2. Dense

Uses multilingual E5 embeddings and vector similarity.

For cross-lingual queries, the application uses the translated query for
dense retrieval. This is the main semantic retrieval approach.

### 3. Hybrid

Combines multiple rankings using weighted Reciprocal Rank Fusion.

For cross-lingual queries, the current configuration uses:

``` text
Translated dense retrieval : weight 2.0
Original-query dense       : weight 0.5
Translated-query BM25      : weight 0.5
```

The weights were selected from the project's controlled 800-query
experiment and should be treated as experimental rather than universally
optimal.

------------------------------------------------------------------------

## Language Support

  Code   Language
  ------ ----------
  `en`   English
  `hi`   Hindi
  `bn`   Bengali
  `te`   Telugu

This gives:

``` text
4 query languages × 4 corpus languages = 16 directions
```

Of these, **12 are cross-lingual** and **4 are same-language**.

------------------------------------------------------------------------

## Dataset

The project uses the **MIRACL multilingual retrieval collection**.

For each supported language, the project contains:

-   **2,863 queries**
-   **5,726 passages**

The dataset is organized so that the same query/passage IDs can be
evaluated across language versions.

The retrieval task is:

> Given a query in one language, retrieve the relevant passage from a
> corpus written in the same or another language.

See:

-   [`docs/dataset.md`](docs/dataset.md)
-   [`docs/methodology.md`](docs/methodology.md)

for the project's dataset and experimental details.

------------------------------------------------------------------------

## NLP and Retrieval Pipeline

### 1. Language Detection

The application detects the query language using Unicode script ranges:

``` text
Devanagari → Hindi
Bengali script → Bengali
Telugu script → Telugu
Other → English
```

This is intentionally lightweight, but it means romanized Hindi such as:

``` text
"kahani ka matlab kya hai"
```

is currently treated as English.

### 2. Query Translation

For cross-lingual retrieval, the query can be translated into the corpus
language using:

``` text
facebook/nllb-200-distilled-600M
```

Translations are cached in memory to reduce repeated inference cost.

### 3. Dense Retrieval

The system uses multilingual E5 embeddings with:

``` text
query: <query>
passage: <passage>
```

prefixes, L2-normalized embeddings, and a maximum sequence length of 256
tokens.

Passage embeddings are computed once and reused during search.

### 4. BM25 Retrieval

The lexical baseline uses Okapi BM25:

``` text
k1 = 1.5
b  = 0.75
```

The tokenizer is designed to preserve Indic Unicode combining marks
rather than relying on a simple ASCII-oriented tokenizer.

### 5. Reciprocal Rank Fusion

The project uses weighted RRF with:

``` text
score(d) = Σ weight / (k + rank)
```

with:

``` text
k = 60
```

### 6. Optional Reranking

The project includes support for:

``` text
BAAI/bge-reranker-v2-m3
```

The reranker can score the top 30 retrieved candidates.

It is **disabled by default** because CPU inference is expensive.

------------------------------------------------------------------------

## Experimental Results

The saved evaluation artifact contains results for all 16 language-pair
combinations.

The main comparison demonstrates the difference between lexical and
semantic retrieval.

### Cross-Lingual Retrieval

Across the twelve cross-lingual directions in the saved evaluation:

  Method       Recall@10    MRR@10
  ---------- ----------- ---------
  BM25           \~1.78%   \~0.012
  Dense E5      \~89.19%   \~0.691
  Hybrid        \~89.18%   \~0.686

The result is consistent with the expected behavior of the two retrieval
paradigms:

-   BM25 struggles when query and passage use different scripts.
-   Multilingual embeddings can match semantic meaning across languages.
-   Adding BM25 through fusion does not automatically improve ranking
    quality.

### Translation Ablation

The controlled 800-query experiment found:

  Strategy                    Recall@10      MRR@10
  ------------------------- ----------- -----------
  Dense, original query           87.5%       0.667
  Dense, translated query         95.1%   **0.806**

The largest gains were observed for directions involving an English
corpus.

Translation therefore acts as a useful bridge between multilingual user
queries and language-specific retrieval collections.

### Important Experimental Note

The repository currently contains:

``` text
experiments/results/retrieval_eval_intfloat__multilingual-e5-small.csv
```

while the current application configuration specifies:

``` text
intfloat/multilingual-e5-base
```

Therefore, the checked-in CSV should be treated as a recorded experiment
artifact rather than evidence that every reported number was generated
by the current configuration. For a strict reproducible benchmark,
regenerate the evaluation after fixing the model configuration.

------------------------------------------------------------------------

## Project Structure

``` text
CrossLingual-Academic-Retrieval/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   │
│   │   ├── routes/
│   │   │   ├── papers.py
│   │   │   └── search.py
│   │   │
│   │   ├── schemas/
│   │   │   └── search.py
│   │   │
│   │   └── services/
│   │       ├── bm25.py
│   │       ├── calibration.py
│   │       ├── data_loader.py
│   │       ├── dense_retrieval.py
│   │       ├── embeddings.py
│   │       ├── evaluation.py
│   │       ├── preprocessing.py
│   │       ├── reranker.py
│   │       ├── retrieval.py
│   │       └── translate.py
│   │
│   ├── data/
│   │   └── download_dataset.py
│   │
│   ├── debug_query.py
│   ├── evaluate_pipeline.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       ├── App.jsx
│       ├── constants.js
│       └── index.css
│
├── experiments/
│   └── results/
│
├── docs/
│   ├── dataset.md
│   ├── methodology.md
│   └── references.md
│
├── public/
│   └── image.png
│
└── README.md
```

------------------------------------------------------------------------

## Tech Stack

### Backend

-   Python
-   FastAPI
-   sentence-transformers
-   scikit-learn
-   SciPy
-   NumPy
-   BM25
-   Hugging Face Transformers

### NLP / ML Models

  Component           Model / Method
  ------------------- ------------------------------------
  Dense retrieval     `intfloat/multilingual-e5-base`
  Translation         `facebook/nllb-200-distilled-600M`
  Optional reranker   `BAAI/bge-reranker-v2-m3`
  Lexical retrieval   Okapi BM25
  Rank fusion         Weighted Reciprocal Rank Fusion

### Frontend

-   React 19
-   Vite
-   React Router

------------------------------------------------------------------------

# Installation

## Requirements

Recommended:

-   Python 3.10+
-   Node.js 18+
-   npm
-   At least 8 GB RAM
-   More RAM is recommended for model loading
-   GPU is strongly recommended for faster model/index creation

CPU execution is possible but can be slow.

------------------------------------------------------------------------

## 1. Clone the Repository

``` bash
git clone https://github.com/srushtisoni08/CrossLingual-Academic-Retrieval
cd CrossLingual-Academic-Retrieval
```

------------------------------------------------------------------------

## 2. Backend Setup

### Create a virtual environment

Windows:

``` powershell
cd backend
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

``` bash
cd backend
python -m venv venv
source venv/bin/activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

### Download the dataset

``` bash
python data/download_dataset.py
```

The dataset is downloaded into:

``` text
backend/data/raw/
```

### Start the backend

``` bash
uvicorn app.main:app --reload
```

Backend:

``` text
http://127.0.0.1:8000
```

------------------------------------------------------------------------

## 3. Frontend Setup

Open another terminal:

``` bash
cd frontend
npm install
npm run dev
```

Frontend:

``` text
http://localhost:5173
```

The Vite development server proxies API requests to the FastAPI backend.

------------------------------------------------------------------------

# Using the Application

Open:

``` text
http://localhost:5173
```

Select:

1.  Query language
2.  Corpus language
3.  Retrieval mode
4.  Number of results

Then enter an academic query.

Example:

``` text
quantum field theory
```

Example API request:

``` text
/api/search?q=quantum+field+theory&corpus_lang=te&mode=hybrid
```

------------------------------------------------------------------------

# API

## Search

``` http
GET /api/search
```

Example:

``` text
/api/search?q=quantum+field+theory&corpus_lang=te&mode=dense
```

Common parameters:

  Parameter       Description
  --------------- ------------------------------
  `q`             Search query
  `corpus_lang`   Target corpus language
  `mode`          `bm25`, `dense`, or `hybrid`
  `top_k`         Number of results

------------------------------------------------------------------------

# Evaluation

The project contains separate evaluation utilities for the retrieval
components.

### Dense Retrieval

``` bash
cd backend
python -m app.services.dense_retrieval
```

### BM25

``` bash
python -m app.services.bm25
```

### Calibration / Off-topic Analysis

``` bash
python -m app.services.calibration
```

### Full Pipeline Evaluation

``` bash
python evaluate_pipeline.py
```

### Single Query Debugging

``` bash
python debug_query.py
```

------------------------------------------------------------------------

## Evaluation Metrics

### Recall@1

Measures the percentage of queries for which the relevant passage
appears at rank 1.

``` text
Recall@1 =
queries with relevant passage at rank 1
----------------------------------------
             total queries
```

### Recall@10

Measures whether the relevant passage appears anywhere in the top 10
results.

``` text
Recall@10 =
queries with relevant passage in top 10
---------------------------------------
             total queries
```

### MRR@10

Mean Reciprocal Rank evaluates how early the relevant passage appears.

``` text
MRR@10 = mean(1 / rank)
```

A query receives zero if its relevant passage is outside the top 10.

------------------------------------------------------------------------

# Configuration

Most model and retrieval settings are controlled from:

``` text
backend/app/config.py
```

Important settings include:

  Setting                  Purpose
  ------------------------ --------------------------------
  `LANGUAGES`              Supported languages
  `EMBEDDING_MODEL`        Dense embedding model
  `EMBEDDING_BATCH_SIZE`   Embedding batch size
  `MAX_SEQ_LENGTH`         Maximum input length
  `TRANSLATE_QUERY`        Enable query translation
  `TRANSLATION_MODEL`      NLLB translation model
  `RERANK`                 Enable cross-encoder reranking
  `RERANKER_MODEL`         Reranking model
  `RERANK_TOP_N`           Number of candidates reranked

------------------------------------------------------------------------

# Performance and Engineering Notes

### First startup

The first startup can be slow because the system needs to:

1.  Download Hugging Face models
2.  Load the embedding model
3.  Load or build retrieval indexes
4.  Encode thousands of passages

The project documentation reports that encoding approximately 5,700
passages per language with E5-base can take around 30 minutes per
language on CPU.

Once indexes are saved under:

``` text
backend/models/
```

later startups are considerably faster.

### Translation caching

Repeated translations are cached in memory.

This reduces latency for repeated queries.

### Reranker

The BGE cross-encoder is disabled by default:

``` python
RERANK = False
```

CPU inference is expensive, so enabling it is more suitable for GPU
systems or offline evaluation.

------------------------------------------------------------------------

# Known Limitations

This is a research/academic project, not a production search engine.

### 1. Script-based language detection

The current detector cannot reliably identify:

-   Romanized Hindi
-   Romanized Bengali
-   Mixed-language queries

### 2. Translation quality

Incorrect translation of technical terms, named entities, or
abbreviations can negatively affect retrieval.

### 3. Benchmark limitations

The evaluation dataset provides one labelled relevant passage per query.
Real search scenarios can have multiple valid answers.

### 4. Corpus limitations

The benchmark corpus is not equivalent to a complete academic database
such as Google Scholar, Semantic Scholar, or a university repository.

### 5. Fusion weights

The current fusion weights were selected using the same 800-query sample
used for the comparison, so those results may be optimistic.

### 6. Language coverage

Only four languages are currently supported:

``` text
English
Hindi
Bengali
Telugu
```

------------------------------------------------------------------------

# Future Improvements

Potential research and engineering directions include:

-   Fine-tune the multilingual embedding model on domain-specific
    academic retrieval data
-   Evaluate all 12 cross-lingual directions with the translation
    pipeline
-   Learn fusion weights instead of manually selecting them
-   Add multilingual cross-encoder reranking
-   Add support for romanized Indian-language queries
-   Add query expansion for academic terminology
-   Improve language detection for mixed-language input
-   Add confidence estimation for retrieved results
-   Evaluate on larger, natively multilingual academic corpora
-   Add semantic caching for repeated searches
-   Use GPU acceleration for faster indexing and reranking
-   Compare additional multilingual embedding models
-   Add user feedback as a retrieval signal

------------------------------------------------------------------------

# Research Documentation

Detailed project documentation is available in:

``` text
docs/dataset.md
docs/methodology.md
docs/references.md
```

The methodology document contains:

-   Retrieval pipeline
-   BM25 configuration
-   Dense retrieval configuration
-   Translation experiments
-   Fusion experiments
-   Evaluation protocol
-   Error analysis
-   Engineering observations
-   Limitations
-   Future work

------------------------------------------------------------------------

# Project Highlights

This project demonstrates that cross-lingual academic retrieval can be
built by combining traditional information retrieval with multilingual
representation learning.

The main experimental observations are:

> **Multilingual dense retrieval is dramatically stronger than BM25 for
> cross-lingual search.**

> **Query translation can substantially improve retrieval, particularly
> when the target corpus is English.**

> **Rank fusion is not automatically beneficial; weak retrieval signals
> can hurt an already strong semantic ranking.**

These findings make the project useful not only as an application but
also as a comparative study of multilingual retrieval strategies.

------------------------------------------------------------------------

## Author

**Srushti Soni**\
B.Tech Computer Engineering\
A. D. Patel Institute of Technology, Anand, Gujarat, India

------------------------------------------------------------------------

## References

The project's references are maintained in:

[`docs/references.md`](docs/references.md)

Key technologies used in the project include MIRACL, multilingual E5,
NLLB-200, BM25, Reciprocal Rank Fusion, and BGE reranking.s