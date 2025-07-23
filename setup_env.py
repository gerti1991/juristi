#!/usr/bin/env python3
"""
Setup script for Albanian Legal RAG System
Helps users configure their environment properly
"""

import os
import shutil
import sys

def setup_environment():
    """Setup the environment for the Albanian Legal RAG System"""
    
    print("🏛️ Albanian Legal RAG System Setup")
    print("=" * 50)
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("📝 Creating .env file from template...")
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file")
            print("⚠️  Please edit .env and add your Groq API key!")
            print("   Get your free API key from: https://console.groq.com/")
        else:
            print("❌ .env.example not found!")
            return False
    else:
        print("✅ .env file already exists")
    
    # Check API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('GROQ_API_KEY')
        if api_key and api_key != 'your_groq_api_key_here':
            print("✅ API key configured")
        else:
            print("⚠️  Please add your Groq API key to .env file")
            return False
    except ImportError:
        print("⚠️  python-dotenv not installed. Run: pip install -r requirements.txt")
        return False
    
    # Check if legal documents exist
    doc_file = os.path.join('legal_documents', 'albanian_legal_rag_documents.json')
    if os.path.exists(doc_file):
        print("✅ Legal documents found")
    else:
        print("⚠️  Legal documents not found")
        print("   The system will still work but with limited functionality")
    
    print("\n🎉 Setup Complete!")
    print("\n🚀 To start the application:")
    print("   streamlit run app.py")
    print("\n💡 To see a demo:")
    print("   python demo.py")
    
    return True

if __name__ == "__main__":
    if setup_environment():
        print("\n✅ Ready to use Albanian Legal RAG System!")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)
