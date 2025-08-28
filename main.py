#!/usr/bin/env python3
"""
Albanian Legal RAG System - Main Entry Point

This script provides a unified entry point for all system operations:
- Processing embeddings
- Running the Streamlit UI
- Starting the API server
- System maintenance tasks
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.juristi.config import config


def setup_logging():
    """Set up logging configuration."""
    config.setup_logging()
    logger = logging.getLogger(__name__)
    return logger


def process_embeddings():
    """Process legal documents and create embeddings."""
    logger = logging.getLogger(__name__)
    logger.info("🔄 Starting embedding processing...")
    
    try:
        from scripts.process_embeddings import main as process_main
        process_main()
        logger.info("✅ Embedding processing completed successfully")
    except Exception as e:
        logger.error(f"❌ Embedding processing failed: {e}")
        sys.exit(1)


def run_ui():
    """Launch the Streamlit UI."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Streamlit UI...")
    
    import subprocess
    import os
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(project_root)
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(project_root / "src" / "juristi" / "ui" / "modern_main.py"),
            "--server.port", str(config.ui.page_title if hasattr(config.ui, 'port') else 8501),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start Streamlit UI: {e}")
        sys.exit(1)


def run_api():
    """Launch the FastAPI server."""
    logger = logging.getLogger(__name__)
    logger.info("🔗 Starting FastAPI server...")
    
    try:
        from scripts.run_api import main as api_main
        api_main()
    except Exception as e:
        logger.error(f"❌ API server failed: {e}")
        sys.exit(1)


def run_telegram_bot():
    """Launch Telegram bot integration."""
    logger = logging.getLogger(__name__)
    logger.info("🤖 Starting Telegram bot...")
    
    try:
        from scripts.telegram_bot import main as telegram_main
        telegram_main()
    except Exception as e:
        logger.error(f"❌ Telegram bot failed: {e}")
        logger.info("💡 Install required packages: pip install python-telegram-bot")
        sys.exit(1)


def validate_system():
    """Validate system configuration and dependencies."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Validating system configuration...")
    
    # Validate configuration
    validation_result = config.validate()
    
    if not validation_result["valid"]:
        logger.error("❌ Configuration validation failed:")
        for error in validation_result["errors"]:
            logger.error(f"  • {error}")
        sys.exit(1)
    
    if validation_result["warnings"]:
        logger.warning("⚠️ Configuration warnings:")
        for warning in validation_result["warnings"]:
            logger.warning(f"  • {warning}")
    
    # Check dependencies
    try:
        import streamlit
        import langchain
        import chromadb
        import sentence_transformers
        logger.info("✅ Core dependencies available")
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.error("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check embeddings
    try:
        from src.juristi.core.rag_engine import AlbanianLegalRAG
        logger.info("✅ RAG engine importable")
    except ImportError as e:
        logger.error(f"❌ RAG engine import failed: {e}")
        sys.exit(1)
    
    logger.info("✅ System validation completed successfully")


def rebuild_index():
    """Rebuild the document index."""
    logger = logging.getLogger(__name__)
    logger.info("🔄 Rebuilding document index...")
    
    try:
        from scripts.rebuild_index import main as rebuild_main
        rebuild_main()
        logger.info("✅ Index rebuild completed successfully")
    except Exception as e:
        logger.error(f"❌ Index rebuild failed: {e}")
        sys.exit(1)


def show_status():
    """Show system status and configuration."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("🏛️  Albanian Legal RAG System Status")
    print("="*60)
    
    # System info
    print(f"📍 Project Root: {project_root}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Configuration
    print("\n📋 Configuration:")
    print(f"  • Embedding Provider: {config.model.embedding_provider}")
    print(f"  • Device: {config.model.device}")
    print(f"  • Database Path: {config.database.chroma_db_path}")
    print(f"  • Documents Path: {config.database.documents_path}")
    
    # Check paths
    chroma_exists = Path(config.database.chroma_db_path).exists()
    docs_exist = Path(config.database.documents_path).exists()
    
    print(f"\n📂 Paths Status:")
    print(f"  • ChromaDB: {'✅ Found' if chroma_exists else '❌ Missing'}")
    print(f"  • Documents: {'✅ Found' if docs_exist else '❌ Missing'}")
    
    # API Keys
    has_google = config.has_google_credentials()
    print(f"\n🔑 Credentials:")
    print(f"  • Google API: {'✅ Available' if has_google else '❌ Missing (will use fallbacks)'}")
    
    # Validation
    validation = config.validate()
    print(f"\n✅ Validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
    
    if validation["errors"]:
        print("❌ Errors:")
        for error in validation["errors"]:
            print(f"  • {error}")
    
    if validation["warnings"]:
        print("⚠️ Warnings:")
        for warning in validation["warnings"]:
            print(f"  • {warning}")
    
    print("\n" + "="*60)


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Albanian Legal RAG System - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py status              # Show system status
  python main.py validate            # Validate configuration
  python main.py process             # Process embeddings
  python main.py ui                  # Start Streamlit UI
  python main.py api                 # Start API server
  python main.py telegram            # Start Telegram bot
  python main.py rebuild             # Rebuild index
        """
    )
    
    parser.add_argument(
        "command",
        choices=["status", "validate", "process", "ui", "api", "telegram", "rebuild"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        config.logging.level = "DEBUG"
    
    logger = setup_logging()
    
    # Execute command
    try:
        if args.command == "status":
            show_status()
        elif args.command == "validate":
            validate_system()
        elif args.command == "process":
            process_embeddings()
        elif args.command == "ui":
            run_ui()
        elif args.command == "api":
            run_api()
        elif args.command == "telegram":
            run_telegram_bot()
        elif args.command == "rebuild":
            rebuild_index()
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
