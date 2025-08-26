# ⚖️ Juristi AI - Albanian Legal Assistant

<div align="center">
  <img src="https://via.placeholder.com/800x200/1f4e79/ffffff?text=Juristi+AI+-+Albanian+Legal+Assistant" alt="Juristi AI Banner" style="border-radius: 15px; margin: 20px 0;"/>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜-green?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector-orange?style=for-the-badge)
![Google AI](https://img.shields.io/badge/Google_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)

**🚀 Advanced AI-Powered Legal Research System for Albanian Law**

*Powered by Modern RAG Architecture & Multi-Provider Embeddings*

</div>

---

## 🌟 **Overview**

**Juristi AI** is a state-of-the-art Retrieval-Augmented Generation (RAG) system specifically designed for Albanian legal research. Built with cutting-edge AI technology, it provides intelligent legal assistance by analyzing comprehensive Albanian legal documents and generating contextual responses.

### 🎯 **Key Features**

<div align="center">
  <table>
    <tr>
      <td align="center" width="25%">
        <img src="https://via.placeholder.com/120x120/28a745/ffffff?text=🧠" alt="AI Brain"/>
        <h4>Intelligent Analysis</h4>
        <p>Advanced AI synthesis from multiple legal documents</p>
      </td>
      <td align="center" width="25%">
        <img src="https://via.placeholder.com/120x120/17a2b8/ffffff?text=⚡" alt="Fast"/>
        <h4>Lightning Fast</h4>
        <p>Optimized vector search with instant responses</p>
      </td>
      <td align="center" width="25%">
        <img src="https://via.placeholder.com/120x120/ffc107/000000?text=📚" alt="Comprehensive"/>
        <h4>Comprehensive</h4>
        <p>19+ Albanian legal codes and regulations</p>
      </td>
      <td align="center" width="25%">
        <img src="https://via.placeholder.com/120x120/dc3545/ffffff?text=🔒" alt="Reliable"/>
        <h4>Enterprise Ready</h4>
        <p>Robust architecture with fallback systems</p>
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
| **Frontend** | Streamlit | Modern web interface |
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
- 4GB+ RAM recommended
- Google AI API Key (optional, has fallbacks)

### Installation

```bash
# Clone the repository
git clone https://github.com/gerti1991/juristi.git
cd juristi

# Install dependencies
pip install -r requirements.txt

# Set up environment (optional)
cp .env.example .env
# Edit .env with your API keys
```

### Usage

**Step 1: Process Documents (First Time Only)**
```bash
python scripts/process_embeddings.py
```

**Step 2: Launch the Application**
```bash
streamlit run src/juristi/ui/modern_main.py
```

**Step 3: Start Querying!**
- Open your browser to `http://localhost:8501`
- Ask questions in Albanian or English
- Get comprehensive legal analysis

---

## 💡 **Example Queries**

<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">

**🔍 Criminal Law**
- *"Sa është denimi për vrasje në Kodin Penal Shqiptar?"*
- *"What are the penalties for theft in Albanian law?"*

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

---

## 🎨 **Features in Detail**

### 🧠 **Intelligent Document Synthesis**
- Combines information from multiple legal sources
- Provides contextual citations and references
- Generates comprehensive legal analysis

### ⚡ **Multi-Provider Embedding System**
- **Primary**: Google Generative AI (768-dim → normalized to 384-dim)
- **Fallback 1**: BGE Small English (384-dim) 
- **Fallback 2**: Sentence Transformers (384-dim)
- Automatic provider switching on quota/error

### 📊 **Incremental Processing**
- Smart document tracking system
- Only processes new or modified PDFs
- Persistent storage with ChromaDB
- Resumable processing after interruptions

### 🌐 **Modern Web Interface**
- Clean, responsive Streamlit UI
- Real-time query processing
- Source document citations
- Conversation memory
- Professional legal styling

---

## 📈 **Performance Metrics**

<div align="center">
  <table>
    <tr>
      <td align="center">
        <h3>📄 2,923</h3>
        <p>Document Pages Processed</p>
      </td>
      <td align="center">
        <h3>🗂️ 9,144</h3>
        <p>Searchable Text Chunks</p>
      </td>
      <td align="center">
        <h3>📚 19</h3>
        <p>Legal Document Collections</p>
      </td>
      <td align="center">
        <h3>⚡ ~2s</h3>
        <p>Average Query Response Time</p>
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
│   ├── 🎨 ui/               # Streamlit interface
│   ├── 📊 data/             # Data processing utilities
│   └── 🔗 api/              # REST API interface
├── 📁 scripts/              # Utility scripts
│   ├── ⚙️ process_embeddings.py  # Document processing
│   └── 🚀 run_streamlit.py       # App launcher
├── 📁 legal_documents/      # Legal document storage
│   └── 📄 pdfs/            # Source PDF files (19 documents)
├── 📁 chroma_db/           # Vector database
│   ├── 🗃️ chroma.sqlite3   # ChromaDB storage
│   └── 📋 document_index.json  # Processing tracker
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
