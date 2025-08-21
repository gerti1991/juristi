#!/usr/bin/env python3
"""
Streamlit Application Launcher

Launch the Albanian Legal RAG system with Streamlit interface.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Main entry point for Streamlit app."""
    try:
        print("🚀 Starting Albanian Legal RAG Streamlit Interface...")
        
        # Get paths
        project_root = Path(__file__).parent.parent
        streamlit_main = project_root / "src" / "juristi" / "ui" / "main.py"
        
        # Ensure the main UI file exists
        if not streamlit_main.exists():
            print(f"❌ Streamlit main file not found: {streamlit_main}")
            sys.exit(1)
        
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root / "src")
        
        # Run streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_main),
            "--server.headless", "false",
            "--server.runOnSave", "true"
        ]
        
        print(f"🔄 Launching: {' '.join(cmd)}")
        print(f"📂 Working directory: {project_root}")
        print(f"🌐 Once started, open: http://localhost:8501")
        
        # Launch streamlit
        subprocess.run(cmd, cwd=project_root, env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Streamlit app...")
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
