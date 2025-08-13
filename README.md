# Albanian Legal RAG (FastAPI + Gemini)

Production-grade Retrieval-Augmented Generation for Albanian legal documents. Uses Google text-embedding-004 for embeddings, Gemini 2.x for generation, plus hybrid retrieval and token-efficient context packing.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io)

## Features

- Google embeddings (text-embedding-004) with cache and auto-invalidation
- Hybrid retrieval: dense + TF‑IDF, MMR diversification, neighbor expansion
- Super-chunking (large overlapping chunks) for better cross-paragraph recall
- Extractive context packing to reduce tokens and latency
- Lightweight session memory (session_id) injected into prompts
- Gemini 2.x Flash for answers, with graceful fallback when unavailable
- FastAPI service with CORS enabled; optional Streamlit UI

## Quick start (Windows PowerShell)

1) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

2) Set environment variables (required + useful knobs)

```powershell
$env:GEMINI_API_KEY = "<your_gemini_api_key>"
# Optional tuning
$env:RAG_SUPERCHUNK_CHARS = "4500"        # large chunk size (chars)
$env:RAG_SUPERCHUNK_OVERLAP = "500"        # overlap between super-chunks
$env:RAG_CONTEXT_PACKING = "1"             # extractive packing on/off
$env:MAX_CONTEXT_LENGTH = "4000"           # packing budget (chars)
$env:RAG_HYBRID_ALPHA = "0.5"              # dense/sparse blend
$env:RAG_MMR_LAMBDA = "0.5"                # diversity strength
$env:RAG_NEIGHBOR_EXPANSION = "1"          # include nearby chunks
$env:SIMILARITY_THRESHOLD = "0.15"         # base threshold
```

3) Start the API (no auto-reload on Python 3.13)

```powershell
./run_api.bat 8000
```

4) Call the API

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/search" -Method POST -ContentType "application/json" -Body '{
  "query": "Sa ditë pushimi kam sipas Kodit të Punës?",
  "top_k": 3,
  "session_id": "demo-session",
  "mode": "hybrid"
}' | ConvertTo-Json -Depth 5
```

Optional UI

```powershell
streamlit run app.py
```

## API

- GET /health → { status, embeddings_ready, documents, provider }
- POST /search → body { query, top_k?, session_id?, mode? } returns { response, documents[], session_id }

Mode values: "hybrid" | "embedding" | "sparse". The API compacts context automatically when packing is enabled.

## Data and caching

- Legal PDFs live in `legal_documents/pdfs/`; processed chunks: `legal_documents/processed/`
- RAG JSON: `legal_documents/pdf_rag_documents.json`
- Embedding cache: `document_embeddings_cache_google.json` (auto-invalidates when chunk knobs change)

## Project layout

```
ai.py                 # Gemini/Groq client wrapper
api.py                # FastAPI server (CORS enabled)
app.py                # Core RAG engine + optional Streamlit UI
scraper.py            # Scraper and processing helpers
config.py             # Legacy config (not required for API path)
requirements.txt      # Python dependencies
run_api.bat           # Windows launcher for API
legal_documents/      # PDFs and processed artifacts
```

## Tuning tips

- Increase RAG_SUPERCHUNK_CHARS to improve cross-section recall; cache re-builds automatically
- Keep RAG_CONTEXT_PACKING=1 to cut tokens and speed up Gemini calls
- Adjust RAG_HYBRID_ALPHA for dense vs sparse weighting; 0.5 is a good start
- MMR and neighbor expansion improve diversity and continuity

## Legal disclaimer

Kjo përgjigje është vetëm për qëllime informuese dhe nuk përbën këshillë ligjore. Konsultohuni gjithmonë me një jurist të kualifikuar për çështje specifike.
