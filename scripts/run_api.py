#!/usr/bin/env python3
"""
FastAPI Server Launcher

Launch the Albanian Legal RAG system with FastAPI REST API.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    import uvicorn
    from juristi.api import app
    
    def main():
        """Main function to start the FastAPI server."""
        print("🚀 Starting Albanian Legal RAG API Server...")
        print("📍 API will be available at: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔄 Interactive API: http://localhost:8000/redoc")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # Set to True for development
            log_level="info"
        )
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all dependencies are installed:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting API server: {e}")
    sys.exit(1)
