#!/usr/bin/env python3
"""
System Setup and Configuration Script

Initialize the Albanian Legal RAG system, download models, 
and prepare the environment for first use.
"""

import sys
import os
from pathlib import Path
import subprocess

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'sentence_transformers', 'numpy', 
        'scikit-learn', 'torch', 'requests', 'beautifulsoup4',
        'groq', 'google-generativeai', 'PyPDF2', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        "legal_documents",
        "legal_documents/pdfs", 
        "legal_documents/processed",
        "logs"
    ]
    
    base_path = Path(__file__).parent.parent
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   📂 {directory}")
    
    print("✅ Directories created successfully!")

def check_environment():
    """Check environment variables and configuration."""
    print("🔧 Checking environment configuration...")
    
    env_file = Path(__file__).parent.parent / ".env"
    
    required_vars = [
        'GROQ_API_KEY',
        'GOOGLE_API_KEY'
    ]
    
    if not env_file.exists():
        print("⚠️ No .env file found. Creating sample .env file...")
        with open(env_file, 'w') as f:
            f.write("""# Albanian Legal RAG System Configuration
# Copy this file to .env and fill in your API keys

# Groq API Key (for LLM responses)
GROQ_API_KEY=your_groq_api_key_here

# Google AI API Key (for embeddings)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: HuggingFace API Key
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# System Configuration
RAG_QUICK_START=true
RAG_MODE=traditional
""")
        print("📝 Sample .env file created. Please add your API keys.")
        return False
    
    print("✅ Environment file found!")
    return True

def download_models():
    """Download required ML models."""
    print("📥 Downloading required models...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Download multilingual model
        print("   🔄 Downloading sentence-transformers model...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("   ✅ Sentence transformer model ready!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error downloading models: {e}")
        return False

def initialize_system():
    """Initialize the RAG system to test everything works."""
    print("🚀 Initializing Albanian Legal RAG system...")
    
    try:
        from juristi.core.rag_engine import AlbanianLegalRAG
        from juristi.core.llm_client import CloudLLMClient
        
        # Test RAG system initialization
        print("   🔄 Testing RAG engine...")
        rag = AlbanianLegalRAG(quick_start=True)
        print("   ✅ RAG engine initialized!")
        
        # Test LLM client
        print("   🔄 Testing LLM client...")
        llm = CloudLLMClient()
        print("   ✅ LLM client initialized!")
        
        print("✅ System initialization successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error initializing system: {e}")
        return False

def main():
    """Main setup function."""
    print("🏛️ Albanian Legal RAG System Setup")
    print("=" * 50)
    
    success = True
    
    # Check dependencies
    if not check_dependencies():
        success = False
    
    # Setup directories
    setup_directories()
    
    # Check environment
    if not check_environment():
        success = False
    
    # Download models
    if success and not download_models():
        success = False
    
    # Initialize system
    if success and not initialize_system():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your API keys to the .env file")
        print("2. Run the Streamlit app: python scripts/run_streamlit.py")  
        print("3. Or run the API server: python scripts/run_api.py")
    else:
        print("❌ Setup completed with errors!")
        print("Please resolve the issues above and run setup again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
