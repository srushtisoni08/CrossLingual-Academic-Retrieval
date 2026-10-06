# Methodology

## Task
Given a query in one language (English, Hindi, Bengali or Telugu), retrieve the relevant passage from a corpus in the same or a different language. There are 16 query → corpus pairs; the 12 with different languages are the cross-lingual ones.

## Retrieval pipeline
1. **Language detection.** The query language is guessed from its Unicode script (Devanagari → Hindi, Bengali, Telugu; anything else → English).
2. **Query translation (cross-lingual only).** The query is translated into the corpus language with NLLB-200 (distilled 600M). BM25 cannot match across scripts, so without this step it returns nothing for cross-lingual queries.
3. **Dense retrieval.** Passages are embedded once with `multilingual-e5` (prefixes `query:` / `passage:`, unit-normalised), so search is a single matrix multiplication. Both the original and the translated query are searched.
4. **BM25.** Okapi BM25 over a sparse matrix, with a tokenizer that keeps Indic combining marks and removes small stopword lists.
5. **Fusion.** Weighted Reciprocal Rank Fusion (k = 60) with dense weighted 2 and BM25 weighted 1.
6. **Reranking.** A cross-encoder (`bge-reranker-v2-m3`) reads the original query with each of the top 30 candidates and reorders them.

Steps 2 and 6 can be switched off in `config.py`.

## Metrics
For each query, the single labelled positive passage is the correct answer.
- **Recall@10:** fraction of queries whose positive passage is in the top 10.
- **MRR@10:** mean of 1 / rank of the positive passage (0 if outside the top 10).

## Baseline results (embeddings only, dense retrieval)
Command: `python -m app.services.dense_retrieval`. Values are Recall@10 / MRR@10.

| Query ↓ / Corpus → | en | hi | bn | te |
|---|---|---|---|---|
| en | 0.994 / 0.878 | 0.951 / 0.761 | 0.920 / 0.728 | 0.917 / 0.711 |
| hi | 0.891 / 0.691 | 0.973 / 0.832 | 0.935 / 0.760 | 0.916 / 0.727 |
| bn | 0.818 / 0.606 | 0.911 / 0.726 | 0.965 / 0.815 | 0.885 / 0.693 |
| te | 0.782 / 0.579 | 0.896 / 0.702 | 0.883 / 0.690 | 0.964 / 0.813 |

Model: `multilingual-e5-small`.

## Effect of a larger embedding model
Replacing e5-small with `multilingual-e5-base` changed MRR@10 by less than ±0.011 in every pair (for example en → hi 0.761 → 0.760, te → en 0.579 → 0.589, hi → hi 0.832 → 0.838). Model size is therefore not what limits cross-lingual quality.

| Query ↓ / Corpus → | en | hi | bn | te |
|---|---|---|---|---|
| en | 0.997 / 0.883 | 0.953 / 0.760 | 0.927 / 0.723 | 0.914 / 0.709 |
| hi | 0.893 / 0.683 | 0.980 / 0.838 | 0.938 / 0.760 | 0.918 / 0.729 |
| bn | 0.815 / 0.610 | 0.922 / 0.734 | 0.969 / 0.821 | 0.892 / 0.697 |
| te | 0.798 / 0.589 | 0.899 / 0.707 | 0.887 / 0.688 | 0.964 / 0.815 |

## Findings
- **BM25 is the best baseline for same-language search only.** Across scripts it is near zero (Recall@10 below 0.05), which is why dense retrieval and query translation are needed.
- **Same-language search beats cross-language.** For a given query language, MRR drops by about 0.07 to 0.22 when the corpus is in another language, and the drop is largest when the corpus is English and the query is Bengali or Telugu.
- **Hybrid does not help cross-lingually.** In earlier experiments (e5-small), fusing BM25 with dense search was slightly worse than dense alone for cross-lingual pairs, because BM25 adds noise there. Translating the query first is what gives BM25 something useful to match.
- **Raw similarity is unreliable for rejecting off-topic queries across languages.** Cross-language similarity for real queries (median about 0.80 to 0.83) overlaps with off-topic probes (maximum about 0.82). A threshold on top-1 similarity would reject 43 to 94% of valid cross-lingual queries. The gap between the top-1 and top-10 scores separates them better (AUC about 0.90 to 0.92 cross-lingual, 0.95 to 0.99 same-language).

## Cross-lingual improvements (translation + reranking)
Added after observing poor results for an English query against the Hindi corpus ("disney tales"): the top results had similarity scores around 0.78, inside the range of off-topic queries, while the relevant Mickey Mouse passages ranked 6th and 7th. The two stages address this by moving the query into the corpus language and judging query and passage jointly.

<!-- TODO: add the evaluation of the full pipeline (translation + reranker) here once it has been run:
| Pair | Dense only | + translation | + reranker |
-->

## Limitations
- Only one passage per query is labelled correct, so metrics understate real quality.
- The corpus holds only passages tied to the training queries, not all of Wikipedia.
- Translation quality affects results; NLLB can transliterate named entities in ways that differ from the corpus spelling.
- The reranker and translation model make the first request slow and need about 3 GB of downloads and extra CPU time per query.
- Language detection is script-based, so romanised Hindi (for example "kahani") is treated as English.