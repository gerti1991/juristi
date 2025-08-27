#!/usr/bin/env python3
"""
Modern Albanian Legal RAG System with dependency checking
"""

import sys
import subprocess
import importlib.util

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "langchain-google-genai",
        "langchain-community", 
        "chromadb",
        "langchain"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.replace('-', '_')
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"🔄 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            if install_package(package):
                print(f"✅ Successfully installed {package}")
            else:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def main():
    print("🚀 Setting up Modern Albanian Legal RAG System...")
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        print("❌ Failed to install required dependencies")
        return False
    
    # Now import and run the migration
    try:
        from src.juristi.core.modern_rag_engine import ModernAlbanianLegalRAG
        print("✅ Successfully imported modern RAG engine")
        
        # Create instance to test
        print("🧪 Testing modern RAG system...")
        rag_system = ModernAlbanianLegalRAG(verbose=True)
        print("✅ Modern RAG system initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error setting up modern RAG: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Setup completed successfully!")
    else:
        print("💥 Setup failed!")
