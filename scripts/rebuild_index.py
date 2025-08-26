#!/usr/bin/env python3
"""
Rebuild Document Index Script

This script rebuilds the document_index.json file from existing PDF files.
Use this when embeddings completed successfully but the document index is empty.
"""

import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.juristi.core.rag_engine import AlbanianLegalRAG

def main():
    """Rebuild the document index from existing PDF files."""
    print("🔧 Rebuilding Document Index...")
    print("=" * 50)
    
    # Initialize RAG system in non-UI mode to rebuild index
    rag = AlbanianLegalRAG(ui_only=False, verbose=True)
    
    # Rebuild the document index
    success = rag.rebuild_document_index("legal_documents/pdfs")
    
    if success:
        print("\n" + "=" * 50)
        print("✅ Document index rebuilt successfully!")
        print("🔍 Check chroma_db/document_index.json to verify")
        return 0
    else:
        print("\n" + "=" * 50)
        print("❌ Failed to rebuild document index")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
