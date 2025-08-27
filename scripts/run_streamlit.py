#!/usr/bin/env python3
"""
Launch script for the Modern Albanian Legal RAG System UI

Use: python scripts/run_streamlit.py
Or better: python main.py ui
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration
from src.juristi.config import config


def main():
    """Launch Streamlit with proper configuration."""
    try:
        # Set up logging
        config.setup_logging()
        
        # Build streamlit command
        ui_path = project_root / "src" / "juristi" / "ui" / "modern_main.py"
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(ui_path),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.port", "8501"
        ]
        
        print("🚀 Starting Albanian Legal RAG System UI...")
        print(f"📍 URL: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop")
        
        # Launch streamlit
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start UI: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()