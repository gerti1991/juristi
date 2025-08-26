# 🔧 Juristi AI - Technical Implementation Guide

<div align="center">
  <img src="https://via.placeholder.com/800x150/343a40/ffffff?text=Technical+Guide+-+Step+by+Step+Implementation" alt="Technical Guide Banner" style="border-radius: 10px; margin: 20px 0;"/>
</div>

---

## 📋 **Table of Contents**

1. [System Requirements](#-system-requirements)
2. [Installation & Setup](#-installation--setup)
3. [Architecture Deep Dive](#-architecture-deep-dive)
4. [Document Processing](#-document-processing)
5. [Embedding Configuration](#-embedding-configuration)
6. [User Interface Setup](#-user-interface-setup)
7. [Troubleshooting](#-troubleshooting)
8. [Advanced Configuration](#-advanced-configuration)
9. [Development Guide](#-development-guide)
10. [Production Deployment](#-production-deployment)

---

## 💻 **System Requirements**

### **Minimum Requirements**
- **OS**: Windows 10+, macOS 10.14+, or Linux Ubuntu 18.04+
- **Python**: 3.9+ (3.11 recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for application + 1GB per 1000 legal document pages
- **Network**: Internet connection for AI services

### **Recommended Setup**
- **Python**: 3.11.x for optimal performance
- **RAM**: 8-16GB for large document collections
- **CPU**: Multi-core processor for faster embedding processing
- **Storage**: SSD for better ChromaDB performance

### **API Keys (Optional but Recommended)**
- **Google AI API Key**: For primary embedding provider
- **Alternative providers**: BGE and Sentence Transformers work offline

---

## 🚀 **Installation & Setup**

### **Step 1: Environment Preparation**

```bash
# Clone the repository
git clone https://github.com/your-repo/juristi-ai.git
cd juristi-ai

# Create virtual environment (recommended)
python -m venv juristi_env

# Activate virtual environment
# Windows:
juristi_env\Scripts\activate
# macOS/Linux:
source juristi_env/bin/activate
```

### **Step 2: Dependencies Installation**

```bash
# Install Python dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

### **Step 3: Environment Configuration**

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
notepad .env  # Windows
nano .env     # Linux/macOS
```

**Environment Variables:**
```env
# API Keys (Optional - system has fallbacks)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Model Configuration
EMBEDDING_PROVIDER=google          # google, bge, sentence-transformers
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GEMINI_MODEL=gemini-2.5-flash
DEVICE=auto                        # auto, cpu, cuda

# System Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
BATCH_SIZE=50
VERBOSE=true
```

---

## 🏗️ **Architecture Deep Dive**

### **System Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│ Document        │───▶│ Vector Database │
│   (Legal Codes) │    │ Processing      │    │ (ChromaDB)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        │
┌─────────────────┐    ┌─────────────────┐              │
│ User Interface  │◀───│ RAG Engine      │◀─────────────┘
│ (Streamlit)     │    │ (LangChain)     │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ LLM Service     │
                       │ (Google Gemini) │
                       └─────────────────┘
```

### **Data Flow**

1. **Document Ingestion**: PDFs → Text Extraction → Chunking
2. **Embedding Generation**: Text Chunks → Vector Embeddings → Storage
3. **Query Processing**: User Query → Vector Search → Context Retrieval
4. **Response Generation**: Context + Query → LLM → Formatted Response

---

## 📚 **Document Processing**

### **Step 1: Prepare Your Documents**

Place your PDF documents in the `legal_documents/pdfs/` directory:

```
legal_documents/
└── pdfs/
    ├── Kodi_Civil_2023.pdf
    ├── Kodi_Penal_1995.pdf
    ├── Kodi_Pune_2024.pdf
    └── ... (other legal documents)
```

### **Step 2: Run Document Processing**

```bash
# Interactive mode (recommended for first-time setup)
python scripts/process_embeddings.py

# Command-line mode with specific provider
python scripts/process_embeddings.py --embedding-provider bge

# Force reprocessing of all documents
python scripts/process_embeddings.py --force-reprocess

# Custom document path
python scripts/process_embeddings.py --documents-path /path/to/your/pdfs
```

### **Processing Output**

The system will:
1. **Scan Documents**: Check for new or modified PDFs
2. **Extract Text**: Convert PDF pages to searchable text
3. **Create Embeddings**: Generate vector representations
4. **Store Data**: Save to ChromaDB with persistent indexing

**Expected Output:**
```
🚀 Starting Albanian Legal RAG Embedding Processing
📁 Documents path: legal_documents/pdfs
🔧 Embedding provider: bge
📄 Found 19 PDF files to check
📖 Processing new PDF: Kodi_Penal_1995.pdf
✅ Extracted 106 pages from Kodi_Penal_1995.pdf
📊 Created 9144 document chunks
✅ Successfully processed documents for embeddings
```

---

## 🎯 **Embedding Configuration**

### **Provider Selection Guide**

| Provider | Best For | Pros | Cons |
|----------|----------|------|------|
| **Google AI** | Production use | Highest quality, multilingual | Requires API key, rate limits |
| **BGE** | Balanced performance | Good quality, no API needed | Moderate resource usage |
| **Sentence Transformers** | Offline/local use | Fully offline, lightweight | Lower quality for legal text |

### **Provider Configuration**

#### **Google AI Setup**
```bash
# Get API key from: https://makersuite.google.com/app/apikey
export GOOGLE_API_KEY="your_api_key_here"

# Configure in .env
EMBEDDING_PROVIDER=google
GOOGLE_EMBEDDING_MODEL=models/embedding-001
```

#### **BGE Setup (No API Key Required)**
```bash
# Set provider
EMBEDDING_PROVIDER=bge

# Model will download automatically on first use
```

#### **Sentence Transformers Setup**
```bash
# Set provider  
EMBEDDING_PROVIDER=sentence-transformers
FALLBACK_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### **Dimension Normalization**

The system automatically normalizes all embeddings to **384 dimensions** using PCA for compatibility across providers.

---

## 🎨 **User Interface Setup**

### **Step 1: Launch the Application**

```bash
# Method 1: Direct launch
streamlit run src/juristi/ui/modern_main.py

# Method 2: Using launcher script
python scripts/run_streamlit.py

# Custom port
streamlit run src/juristi/ui/modern_main.py --server.port 8502
```

### **Step 2: Access the Interface**

1. Open your browser
2. Navigate to `http://localhost:8501`
3. Wait for the system to initialize
4. Start querying!

### **Interface Features**

#### **Main Query Interface**
- **Query Input**: Multi-language support (Albanian/English)
- **Response Display**: Formatted legal analysis
- **Source Attribution**: Document citations with page numbers
- **Conversation History**: Context-aware follow-up questions

#### **Sidebar Options**
- **System Status**: Embedding provider, document count
- **Query Settings**: Search parameters, response length
- **Document Management**: Processing status, index information

---

## 🛠️ **Troubleshooting**

### **Common Issues & Solutions**

#### **Issue: Import Errors**
```bash
ModuleNotFoundError: No module named 'src'
```
**Solution:**
```bash
# Ensure you're in the project root directory
cd juristi-ai
python scripts/process_embeddings.py
```

#### **Issue: API Quota Exceeded**
```
Error 429: You exceeded your current quota
```
**Solution:**
- System automatically falls back to BGE embeddings
- Check your Google AI API quota
- Consider using BGE provider: `--embedding-provider bge`

#### **Issue: No Documents Found**
```
⚠️ No documents found in legal_documents/pdfs
```
**Solution:**
```bash
# Check document path
ls legal_documents/pdfs/

# Ensure PDFs are in correct location
mkdir -p legal_documents/pdfs
# Copy your PDF files there
```

#### **Issue: ChromaDB Compatibility**
```
❌ ChromaDB compatibility error
```
**Solution:**
```bash
# Clear and rebuild database
python scripts/process_embeddings.py --force-reprocess
```

#### **Issue: Memory Errors**
```
RuntimeError: CUDA out of memory
```
**Solution:**
```bash
# Use CPU instead of GPU
export DEVICE=cpu

# Or reduce batch size
export BATCH_SIZE=10
```

---

## ⚙️ **Advanced Configuration**

### **Performance Tuning**

#### **Batch Processing**
```python
# Adjust batch sizes in .env
BATCH_SIZE=50          # Default for embeddings
CHUNK_SIZE=1000        # Text chunk size
CHUNK_OVERLAP=200      # Overlap between chunks
```

#### **Memory Optimization**
```python
# For large document collections
EMBEDDING_BATCH_SIZE=10    # Smaller batches
DEVICE=cpu                 # Force CPU usage
CLEAR_CACHE=true          # Clear model cache
```

### **Custom Document Sources**

#### **Adding New Documents**
1. Place PDFs in `legal_documents/pdfs/`
2. Run: `python scripts/process_embeddings.py`
3. System automatically detects and processes new files

#### **Document Metadata**
```python
# Automatic metadata extraction
{
    "title": "Document Title",
    "source": "/path/to/document.pdf",
    "page_number": 42,
    "document_type": "pdf",
    "processed_at": "2025-08-26T15:30:00"
}
```

### **Custom Embedding Models**

#### **Adding New Providers**
```python
# In rag_engine.py
def _initialize_custom_embeddings(self):
    from your_embedding_provider import CustomEmbeddings
    
    return CustomEmbeddings(
        model_name="custom-legal-model",
        dimension=384  # Must normalize to 384
    )
```

---

## 👨‍💻 **Development Guide**

### **Project Structure**
```
juristi-ai/
├── src/juristi/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── rag_engine.py      # Main RAG implementation
│   │   └── llm_client.py      # LLM interface
│   ├── ui/
│   │   ├── __init__.py
│   │   └── modern_main.py     # Streamlit interface
│   └── data/
│       ├── __init__.py
│       └── processing.py      # Document processing
├── scripts/
│   ├── process_embeddings.py  # Document processing script
│   └── run_streamlit.py      # Application launcher
├── legal_documents/
│   └── pdfs/                 # Document storage
├── chroma_db/                # Vector database
├── requirements.txt          # Dependencies
└── .env.example             # Environment template
```

### **Key Classes**

#### **AlbanianLegalRAG**
```python
class AlbanianLegalRAG:
    def __init__(self, ui_only=False):
        # Initialization logic
        
    def process_documents_for_embeddings(self):
        # Document processing
        
    def query(self, question):
        # Query handling
```

#### **Usage Modes**
```python
# Full mode (for processing)
rag = AlbanianLegalRAG(ui_only=False)
rag.process_documents_for_embeddings()

# UI mode (for querying)
rag = AlbanianLegalRAG(ui_only=True)
result = rag.query("Your question here")
```

### **Testing**

```bash
# Run basic functionality test
python -c "
from src.juristi.core.rag_engine import AlbanianLegalRAG
rag = AlbanianLegalRAG(ui_only=True)
print('✅ System initialized successfully')
"

# Test document processing
python scripts/process_embeddings.py --documents-path test_docs/
```

---

## 🚀 **Production Deployment**

### **Docker Deployment**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "src/juristi/ui/modern_main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **Environment Setup**
```bash
# Build and run
docker build -t juristi-ai .
docker run -p 8501:8501 juristi-ai
```

### **Cloud Deployment Options**

#### **Streamlit Cloud**
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Configure environment variables
4. Deploy automatically

#### **AWS/GCP/Azure**
- Use container services
- Configure environment variables
- Set up persistent storage for ChromaDB
- Monitor API quotas and usage

---

## 📊 **Monitoring & Maintenance**

### **System Health Checks**

```bash
# Check document index
ls -la chroma_db/document_index.json

# Check embedding database
ls -la chroma_db/chroma.sqlite3

# Verify document count
python -c "
from src.juristi.core.rag_engine import AlbanianLegalRAG
rag = AlbanianLegalRAG(ui_only=True)
print(f'Documents loaded: {len(rag.processed_documents)}')
"
```

### **Updating Documents**

```bash
# Add new documents to pdfs/ folder
cp new_legal_code.pdf legal_documents/pdfs/

# Process only new documents (automatic)
python scripts/process_embeddings.py

# Force full reprocessing if needed
python scripts/process_embeddings.py --force-reprocess
```

---

## 🆘 **Support & Troubleshooting**

### **Getting Help**

1. **Check Logs**: Look for error messages in terminal output
2. **Verify Setup**: Ensure all steps in this guide were followed
3. **Check Dependencies**: Run `pip list` to verify installations
4. **Test Components**: Use the testing commands provided

### **Common Solutions**

- **Restart the application** if embeddings seem outdated
- **Clear ChromaDB** with `--force-reprocess` for major issues
- **Check disk space** - ChromaDB needs space for vector storage
- **Verify PDF files** are readable and not corrupted

### **Performance Tips**

- Use **SSD storage** for better ChromaDB performance
- Increase **RAM** for processing large document collections
- Use **BGE embeddings** for consistent offline performance
- **Batch process** documents during off-peak hours

---

<div align="center">
  <h2>🎯 Ready to Deploy Your Albanian Legal AI Assistant!</h2>
  <p><em>Follow this guide step-by-step for a successful implementation</em></p>
  
  <img src="https://via.placeholder.com/500x80/28a745/ffffff?text=✅+Technical+Setup+Complete" alt="Setup Complete" style="border-radius: 5px; margin: 20px 0;"/>
</div>

---

**Questions?** Refer to the main [README.md](README.md) or reach out to our support team.
