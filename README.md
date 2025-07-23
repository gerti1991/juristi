# 🏛️ Albanian Legal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for Albanian legal documents using modern AI technologies.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

## � Features

- **🔍 Semantic Search**: Search through Albanian legal documents using natural language
- **🤖 AI-Powered Responses**: Get contextual answers powered by Groq's Llama models
- **🇦🇱 Albanian Language Support**: Optimized for Albanian legal terminology with synonym enhancement
- **⚡ Real-time Processing**: Fast document retrieval and response generation
- **🎯 Smart Fallback**: AI provides general legal knowledge when specific documents aren't found
- **💻 User-Friendly Interface**: Clean Streamlit web interface
- **📊 Comprehensive Database**: 251 document chunks from 19 authentic Albanian legal codes

## 📚 Document Collection

The system includes **251 document chunks** from **19 authentic Albanian legal codes**:

### 🏛️ Core Legal Documents
- **Kodi Civil 2023** - 10 chunks (Civil Code)
- **Kodi I Punes Ligj 2024** - 15 chunks (Labor Code)
- **Ligj Nr. 7895 1995 Kodi Penal I Përditësuar** - 13 chunks (Criminal Code)
- **Ligj Nr. 9062 08052003 Kodi I Familjes** - 11 chunks (Family Code)
- **Kodi Zjedhor Perditesim 2025** - 16 chunks (Electoral Code)
- **Ligj 2014 07 31 102, Kodi Doganor I Rsh** - 22 chunks (Customs Code)

### 📋 Additional Legal Documents (13 more)
- Dispozita Zbatuese Të Kodit Doganor (Customs Implementation)
- Kodi Ajror, 2018 (Aviation Code)
- Various specialized legal codes and regulations

## �️ Technology Stack

- **Frontend**: Streamlit with modern UI
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Search**: Cosine similarity with scikit-learn
- **AI Models**: Groq API (Llama 3.1 8B Instant)
- **Language Processing**: Enhanced Albanian query processing
- **Caching**: Intelligent embedding caching system
- **Language**: Python 3.8+

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/albanian-legal-rag.git
cd albanian-legal-rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy `.env.example` to `.env` and add your Groq API key:
```bash
cp .env.example .env
```
Then edit `.env` and add your actual API key:
```bash
GROQ_API_KEY=your_groq_api_key_here  # Get from https://console.groq.com/
```

### 4. Run Demo
```bash
python demo.py
```

### 5. Start Application
```bash
streamlit run app.py
```

Or use the convenience script:
```bash
start.bat  # Windows
```

## 📁 Project Structure

```
juristi/
├── app.py              # Main Streamlit application
├── ai.py               # Cloud AI integration (Groq API)
├── scraper.py          # Document processing and scraping
├── config.py           # Configuration and API keys
├── setup.py            # PDF document processor
├── demo.py             # System demonstration
├── start.bat           # Easy launcher
├── requirements.txt    # Python dependencies
└── legal_documents/    # Document storage
    ├── pdfs/          # Original PDF files (19 documents)
    ├── processed/     # Processed document chunks
    └── pdf_rag_documents.json  # RAG-ready documents
```

## 🔧 System Requirements

### **Python Dependencies**
```
streamlit
sentence-transformers
scikit-learn
groq
requests
beautifulsoup4
PyPDF2
python-docx
```

### **API Keys**
- **Groq API Key** (for cloud AI responses)
- Configured in `config.py`

## 🎯 Features

### ✅ **Legal Domain Coverage**
- **Labor Law** - Employee rights, vacation days, wages
- **Criminal Law** - Penalties, sanctions, legal procedures
- **Civil Law** - Contracts, property rights, civil procedures
- **Family Law** - Marriage, divorce, child custody
- **Business Law** - Company registration, commercial law
- **Constitutional Law** - Fundamental rights, government structure
- **Administrative Law** - Government procedures, public law

### ✅ **AI Capabilities**
- **Semantic Search** - Understands Albanian legal terminology
- **Context-Aware Responses** - Provides relevant legal citations
- **Professional Tone** - Legal-appropriate language and format
- **Multi-Document Analysis** - Searches across all legal sources

### ✅ **Technical Features**
- **Cloud AI Integration** - Groq API with Llama 3 8B model
- **Real-time Processing** - Fast search and response generation
- **Document Caching** - Optimized performance
- **Error Handling** - Robust error management

## 🧠 How It Works

### **RAG Process:**
1. **User Question** → Convert to vector embedding
2. **Document Search** → Find similar content using cosine similarity
3. **Context Building** → Combine question + relevant documents
4. **AI Generation** → Generate professional legal response
5. **Source Citation** → Show which legal documents were used

### **Example Flow:**
```
Query: "Sa dit pushimi kam në punë?"
  ↓
Search: Labor Code 2024 documents
  ↓
AI Response: "Sipas Kodit të Punës të Shqipërisë 2024, 
             çdo punëmarrës ka të drejtë për 28 ditë pushim..."
  ↓
Sources: Kodi i Punës 2024, Article X
```

## 📊 System Statistics

- ✅ **19 Albanian Legal Documents** processed
- ✅ **251 Document Chunks** for semantic search
- ✅ **Cloud AI Integration** with professional responses
- ✅ **Complete Albanian Legal Coverage** achieved

## 🌐 Web Interface

The Streamlit application provides:
- **Clean Query Interface** in Albanian
- **Real-time Search Results** with relevance scores
- **Professional AI Responses** with legal citations
- **Source Document Display** with references
- **System Status Monitoring** and statistics

## 📝 Usage Examples

### **Labor Law Query:**
```
Input: "Sa dit pushimi kam në punë të detyrueshme?"
Output: Professional response citing Labor Code 2024
Sources: Kodi i Punës LIGJ 2024, specific articles
```

### **Criminal Law Query:**
```
Input: "Çfarë sanksionesh ka për vjedhje?"
Output: Detailed penalties from Criminal Code
Sources: Kodi Penal, relevant sections
```

### **Business Law Query:**
```
Input: "Si regjistroj një kompani?"
Output: Step-by-step company registration process
Sources: Business registration laws and procedures
```

## 🎉 Ready for Professional Use

This Albanian Legal RAG system is production-ready and can handle complex legal queries with authentic Albanian legal document citations and professional AI-generated responses.

---
*Built with: Python, Streamlit, Groq API, sentence-transformers, and authentic Albanian legal documents*
