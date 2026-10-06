# Cross-Lingual Academic Information Retrieval

A multilingual academic information retrieval system that supports **English, Hindi, Bengali, and Telugu**. It can retrieve relevant academic passages even when the query and corpus are written in different languages.

![Project UI](public/image.png)

## Features

- 🌐 Cross-lingual search across 4 languages
- 🔎 BM25 lexical retrieval
- 🧠 Multilingual E5 semantic retrieval
- 🔄 NLLB-200 query translation
- 🧩 Weighted Reciprocal Rank Fusion
- 🎯 Optional BGE cross-encoder reranking
- 📊 Recall@1, Recall@10 and MRR@10 evaluation
- ⚡ Translation caching
- 🚀 FastAPI REST API
- 🖥️ React + Vite frontend

## Supported Languages

| Code | Language |
|---|---|
| `en` | English |
| `hi` | Hindi |
| `bn` | Bengali |
| `te` | Telugu |

The system supports **16 query → corpus language combinations**, including **12 cross-lingual directions**.

## Architecture

```text
User Query
    │
    ▼
Language Detection
    │
    ├── Same-language ──────► Retrieval
    │
    └── Cross-lingual
             │
             ▼
        NLLB-200
      Translation
             │
             ▼
    ┌─────────────────┐
    │ Dense Retrieval │
    │      BM25       │
    └────────┬────────┘
             │
             ▼
          RRF Fusion
             │
             ▼
      Optional Reranker
             │
             ▼
         Top-K Results
```

## Tech Stack

**Backend**

- Python
- FastAPI
- sentence-transformers
- scikit-learn
- Hugging Face Transformers
- NumPy / SciPy

**Frontend**

- React 19
- Vite
- React Router

**Models / Methods**

| Component | Model / Method |
|---|---|
| Dense Retrieval | `intfloat/multilingual-e5-base` |
| Translation | `facebook/nllb-200-distilled-600M` |
| Reranking | `BAAI/bge-reranker-v2-m3` |
| Lexical Retrieval | Okapi BM25 |
| Rank Fusion | Weighted RRF |

## Dataset

The project uses the **MIRACL multilingual retrieval collection**.

```text
Languages : English, Hindi, Bengali, Telugu
Queries   : 2,863 per language
Passages  : 5,726 per language
```

Dataset details:

```text
docs/dataset.md
```

## Results

### Cross-Lingual Retrieval

| Method | Recall@10 | MRR@10 |
|---|---:|---:|
| BM25 | ~1.78% | ~0.012 |
| Dense E5 | ~89.19% | ~0.691 |
| Hybrid | ~89.18% | ~0.686 |

### Translation Ablation

| Strategy | Recall@10 | MRR@10 |
|---|---:|---:|
| Original Query | 87.5% | 0.667 |
| Translated Query | **95.1%** | **0.806** |

The experiments show that multilingual dense retrieval performs substantially better than BM25 for cross-lingual search, while query translation can further improve retrieval.

> Evaluation results are based on the saved experiment artifacts. See the detailed documentation for configuration and reproducibility notes.

## Quick Start

### Clone

```bash
git clone https://github.com/srushtisoni08/CrossLingual-Academic-Retrieval
cd CrossLingual-Academic-Retrieval
```

### Backend

```bash
cd backend
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python data/download_dataset.py
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## API

Search endpoint:

```http
GET /api/search
```

Example:

```text
/api/search?q=quantum+field+theory&corpus_lang=te&mode=dense
```

Supported modes:

```text
bm25
dense
hybrid
```

## Project Structure

```text
CrossLingual-Academic-Retrieval/
│
├── backend/
│   ├── app/
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── services/
│   ├── data/
│   ├── evaluate_pipeline.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   └── src/
│
├── experiments/
│   └── results/
│
├── docs/
│   ├── project_documentation.md
│   ├── dataset.md
│   ├── methodology.md
│   └── references.md
│
├── public/
│   └── image.png
│
└── README.md
```

## Evaluation

Run the complete pipeline:

```bash
cd backend
python evaluate_pipeline.py
```

Individual components:

```bash
python -m app.services.dense_retrieval
python -m app.services.bm25
python -m app.services.calibration
python debug_query.py
```

## Documentation

For the complete technical details:

- 📘 [`project_documentation.md`](docs/project_documentation.md) — complete project documentation
- 📊 [`dataset.md`](docs/dataset.md) — dataset details
- 🔬 [`methodology.md`](docs/methodology.md) — retrieval and evaluation methodology
- 📚 [`references.md`](docs/references.md) — references

## Limitations

- Currently supports four languages.
- Romanized Indian-language queries are not reliably detected.
- Translation errors can affect retrieval.
- CPU inference can be slow.
- The benchmark corpus is smaller than real academic search engines.
- Reranking is disabled by default because of its computational cost.

## Future Work

- Support romanized queries
- Improve multilingual language detection
- Fine-tune embeddings for academic retrieval
- Improve cross-encoder reranking
- Learn fusion weights automatically
- Expand language and corpus coverage
- Add query expansion and confidence estimation
- Improve GPU-based indexing and inference

## Author

**Srushti Soni**

B.Tech Computer Engineering  
A. D. Patel Institute of Technology  
Anand, Gujarat, India