# Albanian Legal RAG System

## 🏛️ Professional Modular Architecture

This Albanian Legal Research Assistant has been **completely refactored** from a collection of scripts into a **professional, maintainable modular codebase** with clean architecture and industry best practices.

## 📁 New Project Structure

```
juristi/
├── src/juristi/              # Main package
│   ├── core/                 # Core RAG functionality
│   ├── data/                 # Document processing
│   ├── api/                  # REST API interface  
│   └── ui/                   # User interfaces
├── tests/                    # Unified test suite
├── scripts/                  # Setup & launcher scripts
├── docs/                     # Comprehensive documentation
└── legal_documents/          # Document storage
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup system and download models
python scripts/setup_system.py

# 3. Configure API keys (copy .env.example to .env)
cp .env.example .env
# Edit .env with your API keys

# 4. Launch Streamlit interface
python scripts/run_streamlit.py

# OR launch REST API
python scripts/run_api.py
```

## 📚 Full Documentation

**👉 See [docs/README_MODULAR.md](docs/README_MODULAR.md) for complete documentation including:**

- **🏗️ Architecture Overview** - Modular design and component structure
- **🛠️ Installation & Setup** - Detailed setup instructions
- **🔧 Configuration Options** - RAG modes, search types, LLM providers
- **📖 Usage Examples** - Python API, REST API, and web interface
- **🧪 Testing & Development** - Running tests and contributing
- **🐛 Troubleshooting** - Common issues and solutions
- **📊 Performance** - Optimization and benchmarks

## ⚡ Key Features

- **🔍 Advanced Search**: Traditional, hierarchical, and sentence-window RAG modes
- **🤖 Multi-LLM Support**: Google Gemini, Groq, HuggingFace integration  
- **📚 Comprehensive Processing**: PDF documents + web scraping from Albanian legal sources
- **🌐 Multiple Interfaces**: Streamlit web app + REST API
- **⚡ Optimized Performance**: Sub-3-second startup with lazy loading
- **🏗️ Professional Architecture**: Modular, testable, maintainable codebase

## 🎯 Migration from Legacy Version

The new modular system maintains **full backward compatibility** with existing document formats and configurations. Your existing `legal_documents/` directory and `.env` settings will work without changes.

---

**⚖️ For complete setup and usage instructions, see [docs/README_MODULAR.md](docs/README_MODULAR.md)**
- FastAPI service with CORS enabled; optional Streamlit UI

## Advanced Retrieval Modes

### 1. Hierarchical RAG (`RAG_MODE=hierarchical`)
Two-step retrieval process:
1. **Document Selection**: Search document-level summaries to identify most relevant legal codes
2. **Chunk Retrieval**: Perform detailed search only within selected documents

Benefits: Better synthesis across document sections, reduced noise from irrelevant documents.

### 2. Sentence-Window Retrieval (`RAG_MODE=sentence_window`)
Precision-focused approach:
1. **Sentence Search**: Find the single most relevant sentence across all documents
2. **Context Expansion**: Build context window with N sentences before/after the match

Benefits: Pinpoint accuracy, optimal context size, reduced hallucination.

### 3. Traditional Hybrid (`RAG_MODE=hybrid`) [Default]
Original implementation with dense+sparse retrieval, MMR, and neighbor expansion.

## Quick start (Windows PowerShell)

1) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

2) **NEW**: Process documents with advanced features (optional)

```powershell
# Generate enhanced document structure for advanced retrieval
python setup_advanced.py
```

3) Set environment variables (required + useful knobs)

```powershell
$env:GEMINI_API_KEY = "<your_gemini_api_key>"

# NEW: Advanced RAG mode selection
$env:RAG_MODE = "hierarchical"              # 'hybrid' | 'hierarchical' | 'sentence_window'
$env:RAG_HIERARCHICAL_TOP_DOCS = "2"        # Top documents for hierarchical search
$env:RAG_SENTENCE_WINDOW_SIZE = "5"         # Sentences before/after center

# Optional tuning
$env:RAG_SUPERCHUNK_CHARS = "4500"          # large chunk size (chars)
$env:RAG_SUPERCHUNK_OVERLAP = "500"         # overlap between super-chunks
$env:RAG_CONTEXT_PACKING = "1"              # extractive packing on/off
$env:MAX_CONTEXT_LENGTH = "4000"            # packing budget (chars)
$env:RAG_HYBRID_ALPHA = "0.5"               # dense/sparse blend
$env:RAG_MMR_LAMBDA = "0.5"                 # diversity strength
$env:RAG_NEIGHBOR_EXPANSION = "1"           # include nearby chunks
$env:SIMILARITY_THRESHOLD = "0.15"          # base threshold
```

4) Start the API (no auto-reload on Python 3.13)

```powershell
./run_api.bat 8000
```

5) Call the API with advanced modes

```powershell
# Hierarchical RAG example
Invoke-RestMethod -Uri "http://localhost:8000/search" -Method POST -ContentType "application/json" -Body '{
  "query": "Sa ditë pushimi kam sipas Kodit të Punës?",
  "top_k": 3,
  "session_id": "demo-session",
  "mode": "hierarchical"
}' | ConvertTo-Json -Depth 5

# Sentence-window retrieval example
Invoke-RestMethod -Uri "http://localhost:8000/search" -Method POST -ContentType "application/json" -Body '{
  "query": "Çfarë dënimi ka për plagosje me armë të ftohtë?",
  "top_k": 2,
  "mode": "sentence_window"
}' | ConvertTo-Json -Depth 5
```

Optional UI

```powershell
streamlit run app.py
```

## API

- GET /health → { status, embeddings_ready, documents, provider, rag_mode, advanced_features }
- POST /search → body { query, top_k?, session_id?, mode? } returns { response, documents[], session_id }

Mode values: "hybrid" | "embedding" | "sparse" | "hierarchical" | "sentence_window". The API compacts context automatically when packing is enabled.

### Advanced Mode Details

**Hierarchical Mode**: Set `"mode": "hierarchical"` in API requests. Requires enhanced documents (run `setup_advanced.py`).
- First searches document summaries to identify relevant legal codes
- Then performs detailed search within selected documents only
- Reduces cross-document noise, improves synthesis

**Sentence-Window Mode**: Set `"mode": "sentence_window"` in API requests. Requires enhanced documents.
- Searches individual sentences for maximum precision
- Expands around best match with configurable window size
- Ideal for specific legal definitions and precise citations

## Data and caching

- Legal PDFs live in `legal_documents/pdfs/`; processed chunks: `legal_documents/processed/`
- RAG JSON: `legal_documents/pdf_rag_documents.json`
- Embedding cache: `document_embeddings_cache_google.json` (auto-invalidates when chunk knobs change)

## Project layout

```
ai.py                           # Gemini/Groq client wrapper
api.py                          # FastAPI server (CORS enabled, advanced modes)
app.py                          # Core RAG engine + optional Streamlit UI
scraper.py                      # Scraper and processing helpers
config.py                       # Legacy config (not required for API path)
setup_advanced.py               # NEW: Enhanced document processing
requirements.txt                # Python dependencies
run_api.bat                     # Windows launcher for API
legal_documents/                # PDFs and processed artifacts
  pdf_rag_documents.json        # Traditional format
  pdf_rag_documents_advanced.json  # NEW: Enhanced format with summaries/sentences
```

## Processing Pipeline

1. **Traditional**: PDFs → chunks → embeddings (existing `pdf_rag_documents.json`)
2. **Enhanced**: PDFs → documents + summaries + sentences + chunks (run `setup_advanced.py`)
   - Document summaries for hierarchical search
   - Sentence-level index for precision retrieval
   - Backward compatibility with traditional format

## Tuning tips

- Increase RAG_SUPERCHUNK_CHARS to improve cross-section recall; cache re-builds automatically
- Keep RAG_CONTEXT_PACKING=1 to cut tokens and speed up Gemini calls
- Adjust RAG_HYBRID_ALPHA for dense vs sparse weighting; 0.5 is a good start
- MMR and neighbor expansion improve diversity and continuity

## Legal disclaimer

Kjo përgjigje është vetëm për qëllime informuese dhe nuk përbën këshillë ligjore. Konsultohuni gjithmonë me një jurist të kualifikuar për çështje specifike.
