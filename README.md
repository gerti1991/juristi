# ⚖️ Juristi AI - Albanian Legal Assistant

<div align="center">
  <img src="https://via.placeholder.com/800x200/1f4e79/ffffff?text=Juristi+AI+-+Albanian+Legal+Assistant" alt="Juristi AI Banner" style="border-radius: 15px; margin: 20px 0;"/>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15.5.2-black?style=for-the-badge&logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-19.1.1-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9.2-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜-green?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector-orange?style=for-the-badge)
![Google AI](https://img.shields.io/badge/Google_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)

**🚀 Advanced AI-Powered Legal Research System for Albanian Law**

*Powered by Modern RAG Architecture, Multi-Provider Embeddings & Full-Stack Web Interface*

</div>

---

## 🌟 **Overview**

**Juristi AI** is a state-of-the-art Retrieval-Augmented Generation (RAG) system specifically designed for Albanian legal research. Built with cutting-edge AI technology and modern web architecture, it provides intelligent legal assistance through multiple interfaces including a responsive Next.js web application, REST API, and traditional Streamlit interface.

### 🎯 **Key Features**

<div align="center">
  <table>
    <tr>
      <td align="center" width="16.66%">
        <img src="https://via.placeholder.com/120x120/28a745/ffffff?text=🧠" alt="AI Brain"/>
        <h4>Dual-Mode Analysis</h4>
        <p>Precise answers OR comprehensive legal synthesis</p>
      </td>
      <td align="center" width="16.66%">
        <img src="https://via.placeholder.com/120x120/17a2b8/ffffff?text=⚡" alt="Fast"/>
        <h4>Lightning Fast</h4>
        <p>Optimized vector search with instant responses</p>
      </td>
      <td align="center" width="16.66%">
        <img src="https://via.placeholder.com/120x120/ffc107/000000?text=📚" alt="Comprehensive"/>
        <h4>Comprehensive</h4>
        <p>19+ Albanian legal codes and regulations</p>
      </td>
      <td align="center" width="16.66%">
        <img src="https://via.placeholder.com/120x120/dc3545/ffffff?text=🔒" alt="Reliable"/>
        <h4>Enterprise Ready</h4>
        <p>Professional legal consultation quality</p>
      </td>
      <td align="center" width="16.66%">
        <img src="https://via.placeholder.com/120x120/6f42c1/ffffff?text=🎯" alt="Smart"/>
        <h4>Source Verification</h4>
        <p>Contextual disambiguation & citation accuracy</p>
      </td>
      <td align="center" width="16.66%">
        <img src="https://via.placeholder.com/120x120/fd7e14/ffffff?text=🌐" alt="Multi Interface"/>
        <h4>Multi-Interface</h4>
        <p>Next.js Web App, FastAPI & Streamlit</p>
      </td>
    </tr>
  </table>
</div>

---

## 🏗️ **Architecture**

<div align="center">
  <img src="https://via.placeholder.com/900x400/f8f9fa/333333?text=Architecture+Diagram:+PDFs+→+Processing+→+Vector+DB+→+AI+Query+→+Response" alt="Architecture Diagram" style="border-radius: 10px; border: 2px solid #dee2e6;"/>
</div>

### 🔧 **Technical Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Next.js 15.5.2 + React 19 + TypeScript | Modern responsive web application |
| **API Backend** | FastAPI | High-performance REST API |
| **Alternative UI** | Streamlit | Rapid prototyping interface |
| **AI Framework** | LangChain | RAG orchestration |
| **LLM** | Google Gemini 2.5 Flash | Response generation |
| **Embeddings** | Google AI / BGE / Sentence Transformers | Multi-provider fallback |
| **Vector DB** | ChromaDB | Document storage & retrieval |
| **PDF Processing** | PyPDF2 / PyMuPDF | Document extraction |
| **Dimension Sync** | PCA Normalization | Unified 384-dim vectors |

---

## 📚 **Legal Document Coverage**

Our system includes comprehensive coverage of Albanian legal framework:

<div align="center">
  <table>
    <tr>
      <td><strong>🏛️ Civil Law</strong></td>
      <td><strong>⚖️ Criminal Law</strong></td>
      <td><strong>👥 Family Law</strong></td>
      <td><strong>💼 Labor Law</strong></td>
    </tr>
    <tr>
      <td>• Civil Code 2023<br>• Civil Procedure Code</td>
      <td>• Penal Code 1995<br>• Military Penal Code<br>• Juvenile Justice Code</td>
      <td>• Family Code 2003</td>
      <td>• Labor Code 2024</td>
    </tr>
    <tr>
      <td><strong>🗳️ Electoral Law</strong></td>
      <td><strong>🚢 Maritime & Transport</strong></td>
      <td><strong>📦 Customs & Trade</strong></td>
      <td><strong>✈️ Aviation Law</strong></td>
    </tr>
    <tr>
      <td>• Electoral Code 2025</td>
      <td>• Railway Code<br>• Various transport laws</td>
      <td>• Customs Code<br>• Customs Enforcement</td>
      <td>• Aviation Code 2018</td>
    </tr>
  </table>
</div>

---

## 🚀 **Quick Start**

### Prerequisites
- Python 3.9+
- Node.js 18+ (for Next.js frontend)
- 4GB+ RAM recommended
- Google AI API Key (optional, has fallbacks)

### Installation

```bash
# Clone the repository
git clone https://github.com/gerti1991/juristi.git
cd juristi

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies for Next.js frontend
cd ui-nextjs
npm install
cd ..

# Set up environment (optional)
cp .env.example .env
# Edit .env with your API keys
```

### Usage Options

#### 🌐 **Option 1: Modern Next.js Web App (Recommended)**

**Step 1: Process Documents (First Time Only)**
```bash
python main.py process
```

**Step 2: Start the FastAPI Backend**
```bash
# Method 1: Using main.py
python main.py api

# Method 2: Direct uvicorn command
uvicorn src.juristi.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Step 3: Start the Next.js Frontend**
```bash
cd ui-nextjs
npm run dev
```

**Step 4: Access the Application**
- Frontend: `http://localhost:3000` (Next.js Web App)
- API Documentation: `http://localhost:8000/docs` (FastAPI Interactive Docs)

#### 📊 **Option 2: Traditional Streamlit Interface**

**Step 1: Process Documents (First Time Only)**
```bash
python main.py process
```

**Step 2: Launch the Streamlit Application**
```bash
python main.py ui
```

**Step 3: Access Streamlit**
- Open your browser to `http://localhost:8501`

#### 🔧 **Option 3: API Only (for Integration)**

```bash
# Start FastAPI server only
uvicorn src.juristi.api.main:app --host 0.0.0.0 --port 8000

# Test API endpoints
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Cilat janë kushtet për divorcin?", "mode": "precise"}'
```

### 🚀 **Quick Start Guide**

1. **Install dependencies** (Python + Node.js)
2. **Process legal documents** (`python main.py process`)
3. **Choose your interface**:
   - **Next.js Web App**: Modern, responsive, production-ready
   - **Streamlit**: Quick prototyping and development
   - **FastAPI**: Direct API integration

---

## 💡 **Example Queries**

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">

**🔍 Criminal Law (Complex Analysis Example)**
- *"Sa dënohet nëse lëviz me makinë në krah të kundërt, në gjendje të dehur, bën aksident dhe vret dy persona?"*
  - **📍 Precise Mode**: Direct penalty information from Penal Code
  - **🧠 Analyzed Mode**: Comprehensive analysis combining traffic violations + intoxication + manslaughter

**⚖️ Civil Matters**
- *"Cilat janë kushtet për divorcin sipas Kodit Civil?"*
- *"What are the property rights in marriage?"*

**💼 Labor Relations**
- *"Cilat janë të drejtat e punëtorit në Kodin e Punës?"*
- *"What are the rules for terminating employment?"*

**🏛️ Procedure**
- *"Si funksionon procedura gjyqësore civile?"*
- *"What are the steps in criminal procedure?"*

</div>

### 🎯 **Dual Query Modes**

**📍 PRECISE MODE:**
- Direct answers from specific legal articles
- 15 source documents analyzed
- Focused, targeted responses
- Quick legal reference lookup

**🧠 ANALYZED MODE:**
- Comprehensive legal analysis
- 20+ source documents synthesized
- Multi-element legal reasoning
- Complex case scenario analysis
- Professional legal consultation format

---

## 🎨 **Features in Detail**

### 🧠 **Dual-Mode Query System**
- **📍 Precise Mode**: Direct answers from specific legal sources (15 documents)
- **🧠 Analyzed Mode**: Comprehensive synthesis from multiple sources (20+ documents)
- **Professional Analysis**: 3-step mandatory verification process
- **Source Verification**: Contextual disambiguation and mismatch detection
- **Albanian Legal Standards**: Professional terminology and citation format

### ⚡ **Multi-Provider Embedding System**
- **Primary**: Google Generative AI (768-dim → normalized to 384-dim)
- **Fallback 1**: BGE Small English (384-dim) 
- **Fallback 2**: Sentence Transformers (384-dim)
- **Automatic Switching**: Provider switching on quota/error with seamless fallback

### 📊 **Incremental Processing**
- Smart document tracking system
- Only processes new or modified PDFs
- Persistent storage with ChromaDB
- Resumable processing after interruptions

### 🌐 **Modern Web Interface**
- **Dual-Mode Selection**: Choose between Precise and Analyzed responses
- **Professional Legal Styling**: Mode-specific UI (purple for analyzed, green for precise)
- **Enhanced Source Display**: Up to 20 sources with detailed metadata
- **Real-time Processing**: Live query status and mode indicators
- **Conversation Memory**: Session-based query history
- **Responsive Design**: Works on desktop, tablet, and mobile

---

## 🔌 **API Reference**

The FastAPI backend provides RESTful endpoints for integration with external systems:

### **POST /search**
Search for legal information with precise mode (15 sources).

```json
{
  "query": "Cilat janë kushtet për divorcin?",
  "mode": "precise"
}
```

### **POST /analyse** 
Comprehensive legal analysis with analyzed mode (20+ sources).

```json
{
  "query": "Sa dënohet nëse lëviz me makinë në krah të kundërt në gjendje të dehur?",
  "mode": "analyzed"
}
```

### **Response Format**
```json
{
  "answer": "Detailed legal analysis...",
  "sources": [
    {
      "content": "Legal text excerpt...",
      "metadata": {
        "source": "document_name.pdf",
        "page": 15,
        "relevance_score": 0.95
      }
    }
  ],
  "query_mode": "precise|analyzed"
}
```

### **Interactive API Documentation**
Visit `http://localhost:8000/docs` for Swagger UI documentation with live testing capabilities.

---

## 📈 **Performance Metrics**

<div align="center">
  <table>
    <tr>
      <td align="center">
        <h3>📄 10,295</h3>
        <p>Document Chunks Processed</p>
      </td>
      <td align="center">
        <h3>🗂️ 15-20</h3>
        <p>Sources per Query (Mode Dependent)</p>
      </td>
      <td align="center">
        <h3>📚 19</h3>
        <p>Legal Document Collections</p>
      </td>
      <td align="center">
        <h3>⚡ ~2-5s</h3>
        <p>Response Time (Mode Dependent)</p>
      </td>
    </tr>
    <tr>
      <td align="center">
        <h3>🎯 2</h3>
        <p>Query Modes Available</p>
      </td>
      <td align="center">
        <h3>🧠 3x</h3>
        <p>More Detail in Analyzed Mode</p>
      </td>
      <td align="center">
        <h3>⚖️ 100%</h3>
        <p>Professional Legal Format</p>
      </td>
      <td align="center">
        <h3>🔍 384</h3>
        <p>Embedding Dimensions</p>
      </td>
    </tr>
  </table>
</div>

---

## 🔧 **Configuration**

### Environment Variables

```bash
# Optional - API Keys for enhanced performance
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Model Configuration
EMBEDDING_PROVIDER=google  # google, bge, or sentence-transformers
GEMINI_MODEL=gemini-2.5-flash
DEVICE=auto  # auto, cpu, or cuda
```

### Customization

The system supports extensive customization:
- **Document Sources**: Add your own legal documents
- **Embedding Models**: Switch between providers
- **Chunk Sizes**: Optimize for your content
- **UI Themes**: Customize the interface
- **Languages**: Extend beyond Albanian/English

---

## 📊 **Project Structure**

```
juristi-ai/
├── 📁 src/juristi/           # Core application code
│   ├── 🧠 core/             # RAG engine & AI logic
│   │   ├── rag_engine.py    # Enhanced dual-mode RAG system
│   │   └── llm_client.py    # LLM client management
│   ├── 🎨 ui/               # Streamlit interface
│   │   └── modern_main.py   # Dual-mode web interface
│   ├── 📊 data/             # Data processing utilities
│   ├── 🔗 api/              # REST API interface
│   └── config.py            # Centralized configuration management
├── 📁 scripts/              # Utility scripts
│   ├── ⚙️ process_embeddings.py  # Document processing
│   └── 🚀 run_streamlit.py       # App launcher
├── 📁 tests/                # Test suite
│   ├── test_dual_mode.py    # Dual-mode functionality tests
│   └── test_search_debug.py # Search debugging tools
├── 📁 legal_documents/      # Legal document storage
│   └── 📄 pdfs/            # Source PDF files (19 documents)
├── 📁 chroma_db/           # Vector database
│   ├── 🗃️ chroma.sqlite3   # ChromaDB storage (10,295 chunks)
│   └── 📋 document_index.json  # Processing tracker
├── 🚀 main.py              # Main CLI entry point
├── 📋 requirements.txt      # Python dependencies
├── ⚙️ .env.example         # Environment template
├── 📖 README.md           # This file
└── 🔧 TECHNICAL_GUIDE.md   # Implementation guide
```

---

## 🤝 **Contributing**

We welcome contributions to improve Juristi AI! Please see our [technical documentation](TECHNICAL_GUIDE.md) for development guidelines.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Follow the technical guide for implementation
4. Submit a pull request

---

## 📞 **Support**

<div align="center">

**Need Help?** 

📧 Email: contact@juristi.al  
📱 Phone: +355 XX XXX XXXX  
🌐 Website: www.juristi.al  
📖 Documentation: [Technical Guide](TECHNICAL_GUIDE.md)

</div>

---

<div align="center">
  <h3>🎉 Built with ❤️ for the Albanian Legal Community</h3>
  <p><em>Empowering legal professionals with AI-driven research capabilities</em></p>
  
  <img src="https://via.placeholder.com/600x100/1f4e79/ffffff?text=🏛️+Advancing+Justice+Through+Technology+🚀" alt="Footer Banner" style="border-radius: 10px; margin: 20px 0;"/>
</div>

---

## 📄 **License**

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

**© 2025 Juristi AI. All rights reserved.**
