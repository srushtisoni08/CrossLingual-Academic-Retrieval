# Methodology

## 1. Task
Given a query in one language (English, Hindi, Bengali or Telugu), retrieve the passage that answers it from a corpus in the same or a different language. There are 16 query → corpus pairs. The 12 pairs with different languages are the cross-lingual ones, and they are the focus of this project. Data is described in [dataset.md](dataset.md): 2,863 queries and 5,726 passages per language, parallel across languages by id.

## 2. Retrieval pipeline
```
query ─► language detection ─► [translate to corpus language, cross-lingual only]
              │
              ├─► dense search, original query     (multilingual-e5)
              ├─► dense search, translated query   (same-language search)
              └─► BM25 search, translated query
                         │
                weighted Reciprocal Rank Fusion
                         │
        [optional cross-encoder rerank of the top candidates]
                         │
                    top-k results
```

1. **Language detection.** The query language is guessed from its Unicode script: Devanagari → Hindi, Bengali script → Bengali, Telugu script → Telugu, anything else → English. Romanised Hindi (for example "kahani") is therefore treated as English.
2. **Query translation (cross-lingual queries only).** The query is translated into the corpus language with NLLB-200 (`facebook/nllb-200-distilled-600M`, int8-quantised for CPU, beam size 2). Translations are cached in memory. Same-language queries skip this step.
3. **Dense retrieval.** Passages are embedded once with `multilingual-e5` (`query:` / `passage:` prefixes, L2-normalised, 256 tokens maximum), so search is one matrix multiplication. For cross-lingual queries both the original and the translated query are searched.
4. **BM25.** Okapi BM25 (k1 = 1.5, b = 0.75) over a sparse matrix. The tokenizer keeps Indic combining marks (a plain `\w+` splits Hindi, Bengali and Telugu words at vowel signs) and removes small stopword lists. For cross-lingual queries BM25 is run on the translated query, because BM25 cannot match across scripts.
5. **Fusion.** Weighted Reciprocal Rank Fusion with k = 60: each retriever contributes `weight / (60 + rank)` per passage. Weights currently in the app: dense (original query) 2, dense (translated query) 2, BM25 1 (see section 4.5, which shows these should be changed).
6. **Reranking (optional, off by default).** A cross-encoder (`BAAI/bge-reranker-v2-m3`) reads the original query together with each of the top 30 candidates and reorders them.

Search modes exposed by the API: `bm25`, `dense` and `hybrid`. In `dense` mode BM25 is not run; in `bm25` mode only BM25 is used. Stub passages (body shorter than 50 characters) are removed from results.

## 3. Evaluation protocol
For every query, the one passage labelled relevant in the dataset is the correct answer.
- **Recall@10:** fraction of queries whose correct passage appears in the top 10.
- **MRR@10:** mean of 1 / rank of the correct passage (0 if it is outside the top 10).

`python -m app.services.dense_retrieval` and `python -m app.services.bm25` evaluate all 16 pairs on all 2,863 queries. `evaluate_pipeline.py` evaluates the full cross-lingual pipeline on a random sample (fixed seed) of selected pairs. `debug_query.py` prints what each stage returns for a single query.

## 4. Results

### 4.1 Embedding baseline: multilingual-e5-small (dense, all queries)
Recall@10 / MRR@10.

| Query ↓ / Corpus → | en | hi | bn | te |
|---|---|---|---|---|
| en | 0.994 / 0.878 | 0.951 / 0.761 | 0.920 / 0.728 | 0.917 / 0.711 |
| hi | 0.891 / 0.691 | 0.973 / 0.832 | 0.935 / 0.760 | 0.916 / 0.727 |
| bn | 0.818 / 0.606 | 0.911 / 0.726 | 0.965 / 0.815 | 0.885 / 0.693 |
| te | 0.782 / 0.579 | 0.896 / 0.702 | 0.883 / 0.690 | 0.964 / 0.813 |

### 4.2 Larger embedding model: multilingual-e5-base
| Query ↓ / Corpus → | en | hi | bn | te |
|---|---|---|---|---|
| en | 0.997 / 0.883 | 0.953 / 0.760 | 0.927 / 0.723 | 0.914 / 0.709 |
| hi | 0.893 / 0.683 | 0.980 / 0.838 | 0.938 / 0.760 | 0.918 / 0.729 |
| bn | 0.815 / 0.610 | 0.922 / 0.734 | 0.969 / 0.821 | 0.892 / 0.697 |
| te | 0.798 / 0.589 | 0.899 / 0.707 | 0.887 / 0.688 | 0.964 / 0.815 |

MRR@10 changed by at most 0.011 in any pair (for example en → hi 0.761 → 0.760, te → en 0.579 → 0.589, hi → hi 0.832 → 0.838), and the encoding time roughly tripled. **Model size is therefore not what limits cross-lingual quality.** Both models are kept in the comparison; the app uses e5-base.

### 4.3 Findings from the baseline
- **BM25 works only within a language.** Across scripts it is near zero (Recall@10 below 0.05), because queries and passages share no tokens. Same-language BM25 reaches Recall@10 of 0.84 to 0.90 for Hindi, Bengali and Telugu.
- **Same-language search beats cross-language.** For a given query language, MRR@10 drops by about 0.07 to 0.22 when the corpus is in another language, and the drop is largest when the corpus is English and the query is Bengali or Telugu.
- **Hybrid (BM25 + dense) does not help cross-lingually.** With e5-small, fusion was equal to or slightly worse than dense alone for the cross-lingual pairs, because BM25 returns nothing or noise there. This is what motivated translating the query before running BM25.
- **Similarity scores are not comparable across languages.** Cross-language cosine similarity for real queries has a median of about 0.80 to 0.83, against about 0.88 within one language, and off-topic probes reach up to 0.82. A fixed similarity threshold would reject 43 to 94% of valid cross-lingual queries. The gap between the top-1 and top-10 scores separates real from off-topic queries better (AUC about 0.90 to 0.92 cross-lingual, 0.95 to 0.99 same-language), so no similarity cut-off is used.

### 4.4 Qualitative case studies (translation and fusion)
Single queries, inspected with `debug_query.py`. These are illustrations, not measurements.

| Query → corpus | Translation | What each stage returned |
|---|---|---|
| "quantum field theory" → te | క్వాంటం క్షేత్ర సిద్ధాంతం (correct) | Dense on the original query: the correct passage is not in the top 10. Dense on the translated query: correct passage is #1. BM25 on the translated query: correct passage is #1. Final result: correct passage #2. |
| "disney tales" → hi | डिज्नी कहानियाँ (correct) | Dense on both the original and the translated query returned unrelated passages with scores of 0.77 to 0.82, the same range as off-topic queries. BM25 on the translated query did find Disney passages (The Little Mermaid #2, A Ring of Endless Light #3). In `dense` mode BM25 is not run, so these never reached the final list. |

Two observations follow from these cases:
1. Translating the query rescues cases where the cross-lingual dense search fails, and it makes BM25 usable across scripts.
2. When all dense scores sit at the noise level, rank fusion cannot tell them apart from a real signal, so dense-only mode loses what BM25 found.

### 4.5 Full-pipeline evaluation (query translation and fusion)
`evaluate_pipeline.py` compares retrieval strategies on 200 randomly sampled queries (fixed seed) for each of four cross-lingual pairs. Recall@10 / MRR@10. With 200 queries the standard error of MRR@10 is about 0.025, so differences smaller than about 0.03 should not be over-interpreted.

| Strategy | en → te | hi → en | te → en | All pairs combined |
|---|---|---|---|---|
| Dense, original query | 0.905 / 0.697 | 0.895 / 0.670 | 0.750 / 0.520 | 0.875 / 0.667 |
| **Dense, translated query** | 0.900 / **0.745** | 0.985 / **0.853** | 0.970 / **0.829** | 0.951 / **0.806** |
| BM25, translated query | 0.655 / 0.469 | 0.920 / 0.737 | 0.890 / 0.720 | 0.801 / 0.618 |
| Fusion: dense orig 2 + dense trans 2 (`dense` mode) | 0.940 / 0.748 | 0.950 / 0.795 | 0.930 / 0.681 | 0.948 / 0.758 |
| Fusion: BM25 1 + dense orig 2 + dense trans 2 (`hybrid`) | 0.940 / 0.736 | 0.970 / 0.836 | 0.965 / 0.753 | 0.958 / 0.772 |
| Fusion: dense trans 2 + BM25 1 | 0.885 / 0.674 | 0.980 / 0.849 | 0.960 / 0.816 | 0.938 / 0.762 |
| Fusion: dense orig 1 + dense trans 1 + BM25 1 | 0.925 / 0.708 | 0.995 / 0.830 | 0.975 / 0.776 | 0.964 / 0.763 |

The "all pairs combined" column covers four pairs (800 queries); the en → hi results are included in it but its separate row was not captured here.

Findings:
- **Translating the query is the largest single improvement for queries written in Hindi or Telugu searching the English corpus.** Dense search on the translated query raises MRR@10 from 0.670 to 0.853 (hi → en) and from 0.520 to 0.829 (te → en), and Recall@10 from 0.75 to 0.97 for te → en. For en → te the gain is smaller (0.697 to 0.745). Across all pairs MRR@10 rises from 0.667 to 0.806.
- **The original query adds noise when fused at full weight.** The setting used by the app in `dense` mode (original and translated query, equal weight) is clearly worse than dense search on the translation alone for te → en (MRR 0.681 against 0.829) and hi → en (0.795 against 0.853). The original query does help Recall@10 for en → te (0.940 against 0.900), so it is useful as a lower-weighted safety net but not as an equal partner.
- **BM25 on the translated query is strong into English and weak into Telugu.** MRR@10 is 0.72 to 0.74 for hi → en and te → en, but 0.47 for en → te, most likely because Telugu is agglutinative and words carry many suffixes that exact-token matching misses.
- **Adding BM25 to the fusion did not beat translated-dense alone on MRR@10.** It does increase Recall@10 (best combined value 0.964 for the equal-weight three-way fusion, against 0.951), so it helps find the passage somewhere in the top 10 but does not put it higher.
- **Conclusion for the default configuration:** for cross-lingual queries, dense search on the translated query should be the main signal, with the original query and BM25 at lower weight. The weights below this line were chosen from the table above and are still to be confirmed with the extended weighting comparison in `evaluate_pipeline.py`.

## 5. Engineering notes
- **Reranker cost.** Scoring 30 candidates with `bge-reranker-v2-m3` took about 80 seconds for one search on a CPU-only machine (two batches of about 40 seconds), and requests queued behind each other. The reranker is therefore disabled by default (`RERANK = False`) and is intended for a GPU or offline evaluation.
- **Startup.** The server now starts as soon as the indexes and the embedding model are loaded; the translation model is loaded in a background thread. Running with `HF_HUB_OFFLINE=1` avoids repeated Hugging Face Hub checks, and running without `--reload` avoids reloading every model on each file change.
- **Search latency.** After the translation model is loaded, an English → Hindi search took about 0.6 s the first time and about 0.09 s when the translation was cached.
- **Index build time.** Encoding about 5,700 passages per language with e5-base took about 30 minutes per language on CPU. Indexes are saved to `backend/models/` and reused.

## 6. Limitations
- Only one passage per query is labelled correct, so the metrics understate real quality.
- The corpus contains only passages tied to the training queries (including many hard negatives), not all of Wikipedia, so some topics have few or no good passages.
- The non-English text appears to be translated from English, so results may differ on natively written data.
- Translation quality affects results. NLLB can transliterate named entities in ways that differ from the corpus spelling, and a wrong translation sends both the dense and BM25 searches the wrong way.
- Language detection is script-based and does not handle romanised text or mixed-language queries.
- The fusion weights currently used by the app were set by hand. Section 4.5 shows that they are not the best of the settings tried, and they will be updated.

## 7. Possible future work
- Tune the fusion weights and the default mode from the full-pipeline evaluation.
- Use a GPU to enable the cross-encoder reranker, or a smaller multilingual reranker.
- Fine-tune the embedding model on the training triplets.
- Detect romanised queries and transliterate them.
- Use the top-1 to top-10 score gap to flag low-confidence results in the UI.