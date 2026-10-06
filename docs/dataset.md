# Dataset

## Source
[`nlpai-lab/miracl-multilingual-triplets`](https://huggingface.co/datasets/nlpai-lab/miracl-multilingual-triplets), a triplet version of **MIRACL** (Multilingual Information Retrieval Across a Continuum of Languages). MIRACL is built from Wikipedia: each query is a natural question, and annotators judged Wikipedia passages as relevant or not.

## Languages used
English (`en`), Hindi (`hi`), Bengali (`bn`), Telugu (`te`). Downloaded by `backend/data/download_dataset.py` into `backend/data/raw/<lang>/`.

## Structure
Each row is a triplet with columns `id`, `query`, `positive`, `negative`:
- `query`: a question
- `positive`: a passage judged relevant to the query
- `negative`: a passage judged not relevant (a hard negative)

Only the `train` split is used. After cleaning (empty rows and duplicate ids dropped), each language has exactly **2,863 queries** (checked with `python -m app.services.data_loader`).

## Size
| | en | hi | bn | te |
|---|---|---|---|---|
| Queries | 2,863 | 2,863 | 2,863 | 2,863 |
| Passages | 5,726 | 5,726 | 5,726 | 5,726 |

## How the corpus is built
`data_loader.build_corpus` makes one passage per positive and one per negative, so each language has **5,726 passages** (2 per query). Passage ids are language-independent (`<id>-p`, `<id>-n`), so the same passage has the same `doc_id` in every language. That is what makes cross-lingual evaluation possible: a Hindi query's relevant document can be checked in the Bengali corpus by id.

Passages are `Title\nbody`. Stub passages with a body shorter than 50 characters are hidden from search results.

## Limitations
- **The corpus is small.** It only contains passages attached to training queries, not all of Wikipedia. A topic with no training query may have few or no good passages.
- **One labelled answer per query.** Evaluation counts only the single positive as correct. Other passages may also answer the query but count as misses, so the reported scores understate real quality.
- **Hard negatives are in the corpus.** This makes ranking harder than a random-negative setup, which is good for evaluation but means scores are not comparable to papers that use a full-Wikipedia corpus.
- **Parallel across languages.** The same `id` is the same query and passage in all four languages (for example, "When was quantum field theory developed?" and its Hindi, Bengali and Telugu versions). The non-English text appears to be translated rather than independently written, so cross-lingual scores may differ from those on natively written data, and translation wording affects them.