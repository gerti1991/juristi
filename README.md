# ⚖️ Juristi AI - Albanian Legal Assistant

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15.5.2-black?style=for-the-badge&logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-19.1.1-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜-green?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector-orange?style=for-the-badge)

**🚀 Advanced AI-Powered Legal Research System for Albanian Law**

*Multi-Platform RAG System with Web UI, REST API & Telegram Bot*

</div>

---

## 🌟 Overview

**Juristi AI** is a comprehensive Retrieval-Augmented Generation (RAG) system designed for Albanian legal research. It provides intelligent legal assistance through multiple interfaces, making Albanian legal knowledge accessible via modern web applications, REST APIs, and instant messaging.

## ✨ Features

- 🧠 **Dual-Mode Analysis**: Precise answers OR comprehensive legal synthesis
- ⚡ **Lightning Fast**: Optimized vector search with instant responses  
- 📚 **Comprehensive**: 19+ Albanian legal codes and regulations
- 🌐 **Multi-Platform**: Web UI, REST API, Streamlit, and Telegram bot
- 🔍 **Advanced Search**: Hybrid semantic and keyword search
- 📱 **Mobile-Friendly**: Responsive design across all devices
- 🚀 **Modern Stack**: Next.js 15, React 19, FastAPI, Python 3.9+

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js UI   │    │  Streamlit UI   │    │ Telegram Bot    │
│   (Port 3000)  │    │   (Port 8501)   │    │  (@Juristi_bot) │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────┐
                    │     FastAPI Server     │
                    │      (Port 8000)       │
                    └─────────────┬───────────┘
                                  │
                    ┌─────────────▼───────────┐
                    │    RAG Engine Core     │
                    │  (ChromaDB + AI Models) │
                    └─────────────────────────┘
```

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/yourusername/juristi.git
cd juristi

# Create virtual environment
python -m venv juristi_env
source juristi_env/bin/activate  # On Windows: juristi_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys:
# - GOOGLE_API_KEY=your_google_ai_key
# - TELEGRAM_BOT_TOKEN=your_bot_token (optional)
```

### 3. Process Legal Documents

```bash
# Process and embed legal documents
python main.py process

# Verify system status
python main.py validate
```

### 4. Start Services

#### Option A: Web Interface (Next.js)
```bash
# Start FastAPI backend
python main.py api &

# Start Next.js frontend
cd ui-nextjs
npm install
npm run dev
```

#### Option B: Streamlit Interface
```bash
python main.py ui
```

#### Option C: Telegram Bot
```bash
python main.py telegram
```

## 🖥️ Usage

### Web Interface (Next.js)
- Navigate to `http://localhost:3000`
- Choose between **Precise** or **Analyzed** search modes
- Enter your legal question in Albanian
- Get instant AI-powered answers with source citations

### REST API
- API Documentation: `http://localhost:8000/docs`
- **POST** `/search` - Precise legal search
- **POST** `/analyse` - Comprehensive legal analysis
- **GET** `/status` - System health check

### Telegram Bot
- Find `@Juristi_bot` on Telegram
- Commands:
  - `/start` - Welcome message
  - `/help` - Usage instructions  
  - `/precise [question]` - Quick legal search
  - `/analyze [question]` - Comprehensive analysis
  - Direct messages work too!

## 📁 Project Structure

```
juristi/
├── src/juristi/           # Core application code
│   ├── api/              # FastAPI REST endpoints
│   ├── core/             # RAG engine & AI logic
│   ├── data/             # Data processing utilities
│   ├── integrations/     # External service integrations
│   └── ui/               # Streamlit interface
├── scripts/              # Utility scripts
│   ├── process_embeddings.py
│   ├── run_api.py
│   ├── run_streamlit.py
│   └── telegram_bot.py
├── ui-nextjs/           # Next.js frontend
├── legal_documents/     # Legal document storage
│   └── pdfs/           # Source PDF documents
├── tests/              # Test suites
├── docs/              # Documentation
└── main.py           # Unified entry point
```

## 🔧 Configuration

### Environment Variables (.env)
```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional - Telegram Integration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ADMIN_CHAT_ID=your_chat_id

# Optional - System Settings
RAG_MODEL_NAME=gemini-1.5-flash
EMBEDDING_MODEL_NAME=models/text-embedding-004
CHROMA_PERSIST_DIRECTORY=./chroma_db
LOG_LEVEL=INFO
```

### Advanced Configuration
- Edit `src/juristi/config.py` for detailed system settings
- Customize embedding models, chunk sizes, and retrieval parameters
- Configure document processing pipelines and filters

## 🧪 Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run with development settings
python main.py ui --dev
```

### API Development
```bash
# Start API in reload mode
uvicorn src.juristi.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Next.js Development
```bash
cd ui-nextjs
npm run dev    # Development server
npm run build  # Production build
npm run lint   # Code linting
```

## 📚 Documentation

- [Technical Guide](TECHNICAL_GUIDE.md) - Detailed technical documentation
- [Telegram Setup](docs/TELEGRAM_SETUP.md) - Telegram bot configuration
- [API Documentation](http://localhost:8000/docs) - Interactive API docs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/juristi/issues)
- 💬 Telegram: [@Juristi_bot](https://t.me/Juristi_bot)

---

<div align="center">
  <strong>⚖️ Making Albanian Legal Knowledge Accessible Through AI ⚖️</strong>
</div>
