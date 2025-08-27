# Albanian Legal RAG System - Modular Architecture

## 🏛️ Overview

A comprehensive Albanian Legal Research Assistant powered by Retrieval-Augmented Generation (RAG) technology. This system has been completely refactored from a collection of scripts into a professional, maintainable modular codebase.

## 🚀 Key Features

- **🔍 Advanced Legal Search**: Multiple search modes including traditional, hierarchical, and sentence-window RAG
- **🤖 AI-Powered Responses**: Integration with multiple LLM providers (Google Gemini, Groq, HuggingFace)
- **📚 Comprehensive Document Processing**: Support for PDF documents and web scraping from Albanian legal sources
- **🌐 Multiple Interfaces**: Both Streamlit web interface and REST API
- **⚡ Performance Optimized**: Lazy loading and caching for fast startup times
- **🏗️ Professional Architecture**: Modular, testable, and maintainable codebase

## 📁 Project Structure

```
juristi/
├── src/juristi/                 # Main package
│   ├── __init__.py             # Package entry point
│   ├── core/                   # Core RAG functionality
│   │   ├── rag_engine.py       # Main RAG engine
│   │   └── llm_client.py       # LLM integration
│   ├── data/                   # Data processing
│   │   └── processing.py       # Document processing & scraping
│   ├── api/                    # REST API interface
│   │   └── main.py             # FastAPI application
│   └── ui/                     # User interfaces
│       └── main.py             # Streamlit interface
├── tests/                      # Test suite
│   └── test_suite.py           # Unified tests
├── scripts/                    # Utility scripts
│   ├── setup_system.py         # System setup
│   ├── run_streamlit.py        # Launch Streamlit app
│   └── run_api.py              # Launch API server
├── docs/                       # Documentation
├── legal_documents/            # Document storage
│   ├── pdfs/                   # PDF files
│   └── processed/              # Processed documents
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## 🛠️ Installation & Setup

### 1. Prerequisites

- Python 3.8+ 
- Git
- 4GB+ RAM recommended
- Internet connection for model downloads

### 2. Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd juristi

# Install dependencies
pip install -r requirements.txt

# Run system setup
python scripts/setup_system.py

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env file with your API keys
```

### 3. API Keys Configuration

Create a `.env` file with your API keys:

```env
# Required API Keys
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# System Configuration
RAG_QUICK_START=true
RAG_MODE=traditional
```

## 🚀 Running the System

### Streamlit Web Interface

```bash
# Launch the web interface
python scripts/run_streamlit.py

# Or using Streamlit directly
streamlit run src/juristi/ui/main.py
```

Access at: `http://localhost:8501`

### REST API Server

```bash
# Launch the API server
python scripts/run_api.py

# Or using uvicorn directly
uvicorn src.juristi.api.main:app --reload
```

- API Base: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`
- Interactive API: `http://localhost:8000/redoc`

## 🔧 Configuration Options

### RAG Modes

- **Traditional**: Standard similarity-based retrieval
- **Hierarchical**: Two-stage retrieval (document summaries → chunks)  
- **Sentence Window**: Sentence-level matching with expanded context

### Search Modes

- **Hybrid**: Combines dense (embedding) and sparse (TF-IDF) retrieval
- **Embedding**: Pure vector similarity search
- **Sparse**: Traditional TF-IDF keyword search

### LLM Providers

- **Google Gemini**: High-quality responses (requires Google API key)
- **Groq**: Fast inference (requires Groq API key)
- **HuggingFace**: Local/cloud models (optional API key)

## 📚 Usage Examples

### Python API Usage

```python
from src.juristi.core.rag_engine import AlbanianLegalRAG
from src.juristi.core.llm_client import CloudLLMClient

# Initialize systems
rag = AlbanianLegalRAG(quick_start=True)
llm = CloudLLMClient()

# Search documents
results = rag.search_documents(
    query="Çfarë janë të drejtat e punëtorit?",
    top_k=3,
    mode="hybrid"
)

# Generate AI response
context = "\n".join([doc['content'] for doc in results])
response = llm.get_legal_response(
    user_query="Çfarë janë të drejtat e punëtorit?",
    context=context
)

print(f"AI Response: {response}")
```

### REST API Usage

```bash
# Search documents
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Çfarë janë të drejtat e punëtorit?",
    "top_k": 3,
    "mode": "hybrid"
  }'

# Get AI response
curl -X POST "http://localhost:8000/ai-response" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Çfarë janë të drejtat e punëtorit?",
    "provider": "gemini"
  }'

# Combined search with AI
curl -X POST "http://localhost:8000/combined-search?include_ai=true" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Çfarë janë të drejtat e punëtorit?",
    "top_k": 3
  }'
```

## 🧪 Testing

```bash
# Run the unified test suite
python -m pytest tests/test_suite.py -v

# Run with coverage
python -m pytest tests/test_suite.py --cov=src/juristi

# Run specific test categories
python -m pytest tests/test_suite.py -k "test_search" -v
```

## 📊 Performance

- **Startup Time**: <3 seconds (with quick start mode)
- **Search Speed**: ~0.1-0.5 seconds per query
- **Memory Usage**: ~2-4GB with models loaded
- **Concurrent Users**: Supports multiple simultaneous requests

## 🔧 Development

### Code Structure

- **`core/`**: Business logic and RAG functionality
- **`data/`**: Document processing and scraping
- **`api/`**: REST API endpoints and models
- **`ui/`**: User interface components
- **`tests/`**: Comprehensive test coverage

### Adding New Features

1. Add core logic to appropriate `core/` module
2. Update API endpoints in `api/main.py` 
3. Update UI components in `ui/main.py`
4. Add tests to `tests/test_suite.py`
5. Update documentation

### Code Quality

```bash
# Format code
black src/

# Check style
flake8 src/

# Type checking
mypy src/
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` directory is in Python path
2. **API Key Errors**: Check `.env` file configuration
3. **Memory Issues**: Reduce model size or use quick_start mode
4. **Slow Performance**: Enable embeddings caching

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug mode in RAG
rag = AlbanianLegalRAG(quick_start=False, debug=True)
```

### Performance Optimization

1. Enable Google embeddings for better quality
2. Use hierarchical RAG mode for large document sets
3. Implement document caching
4. Use GPU acceleration if available

## 📝 API Documentation

### Search Endpoint

`POST /search`

```json
{
  "query": "Legal question in Albanian or English",
  "top_k": 3,
  "mode": "hybrid",
  "rag_mode": "traditional", 
  "multi_query": true,
  "session_id": "optional-session-id"
}
```

### Response Format

```json
{
  "success": true,
  "query": "Original query",
  "results": [
    {
      "id": "doc_id",
      "title": "Document title",
      "content": "Document content...",
      "source": "Source reference",
      "similarity_score": 0.85,
      "rank": 1
    }
  ],
  "total_results": 3,
  "search_time": 0.15,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Run system diagnostics: `python scripts/setup_system.py`
4. Create an issue with system details and error logs

## 🔄 Migration from Legacy Version

If upgrading from the original script-based version:

1. Back up your existing `legal_documents/` directory
2. Run `python scripts/setup_system.py`
3. Copy your `.env` file settings
4. Test with `python scripts/run_streamlit.py`

The new modular system maintains full backward compatibility with existing document formats and configurations.

---

**⚖️ Juristi AI - Empowering Albanian Legal Research with AI Technology**
