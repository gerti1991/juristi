#!/usr/bin/env python3
"""
Embedding Processing Script for Albanian Legal RAG System

This script processes documents and creates embeddings separately from the UI.
Run this script before starting the Streamlit UI to prepare the embeddings database.

Usage:
    python scripts/process_embeddings.py [--documents-path path/to/pdfs]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.juristi.core.rag_engine import AlbanianLegalRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to process documents and create embeddings."""
    parser = argparse.ArgumentParser(description='Process documents and create embeddings for Albanian Legal RAG')
    parser.add_argument(
        '--documents-path', 
        type=str, 
        default='legal_documents/pdfs',
        help='Path to directory containing PDF documents (default: legal_documents/pdfs)'
    )
    parser.add_argument(
        '--persist-directory',
        type=str,
        default='chroma_db',
        help='Directory for ChromaDB persistence (default: chroma_db)'
    )
    parser.add_argument(
        '--embedding-provider',
        type=str,
        choices=['google', 'bge', 'sentence-transformers'],
        default=None,
        help='Embedding provider to use (google, bge, sentence-transformers). If not specified, user will be prompted to choose.'
    )
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing of all documents (clear existing embeddings)'
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Choose embedding provider if not specified
    if args.embedding_provider is None:
        print("\n🔧 Choose Embedding Provider:")
        print("1. Google Generative AI (768 dimensions) - Requires GOOGLE_API_KEY")
        print("2. BGE (384 dimensions) - Uses HuggingFace, good for local processing")
        print("3. Sentence Transformers (384 dimensions) - Lightweight, works offline")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                if choice == "1":
                    args.embedding_provider = "google"
                    # Check if API key is available
                    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
                        print("❌ Warning: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set!")
                        print("   Set it with: set GOOGLE_API_KEY=your_api_key")
                        continue_anyway = input("Continue anyway? (y/N): ").strip().lower()
                        if continue_anyway != 'y':
                            return 1
                    break
                elif choice == "2":
                    args.embedding_provider = "bge"
                    break
                elif choice == "3":
                    args.embedding_provider = "sentence-transformers"
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled by user.")
                return 1
    
    # Set environment variable for chosen provider
    os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider
    
    logger.info("🚀 Starting Albanian Legal RAG Embedding Processing")
    logger.info(f"📁 Documents path: {args.documents_path}")
    logger.info(f"💾 Persist directory: {args.persist_directory}")
    logger.info(f"🔧 Embedding provider: {args.embedding_provider}")
    
    try:
        # Initialize RAG system in full mode (not UI-only)
        logger.info("🔧 Initializing RAG system in full mode...")
        rag_system = AlbanianLegalRAG(
            persist_directory=args.persist_directory,
            verbose=True,
            ui_only=False  # Full mode for embedding processing
        )
        
        # Clear existing embeddings if force reprocess is requested
        if args.force_reprocess:
            logger.info("🗑️ Force reprocessing requested - clearing existing embeddings...")
            rag_system._clear_incompatible_vectorstore()
        
        # Process documents
        logger.info("📚 Processing documents for embeddings...")
        success = rag_system.process_documents_for_embeddings(args.documents_path)
        
        if success:
            logger.info("✅ Embedding processing completed successfully!")
            
            # Display summary
            if hasattr(rag_system, 'total_documents') and rag_system.total_documents > 0:
                logger.info(f"📊 Total documents processed: {rag_system.total_documents}")
            
            logger.info("🎉 Embeddings are ready for UI queries!")
            logger.info("You can now run: streamlit run scripts/run_streamlit.py")
            
        else:
            logger.error("❌ Embedding processing failed!")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Error during embedding processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
