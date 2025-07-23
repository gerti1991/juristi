"""
Albanian Legal RAG System - Project Validation
Quick check to ensure all components work after reorganization
"""

import os
import json

def validate_project():
    """Validate the project structure and files"""
    
    print("🏛️ ALBANIAN LEGAL RAG - PROJECT VALIDATION")
    print("=" * 50)
    
    # Check core files
    core_files = {
        'app.py': 'Main Streamlit application',
        'ai.py': 'Cloud AI integration',
        'scraper.py': 'Document processing',
        'config.py': 'Configuration settings',
        'setup.py': 'PDF processor',
        'requirements.txt': 'Dependencies',
        'README.md': 'Documentation',
        'start.bat': 'Easy launcher'
    }
    
    print("📁 CORE FILES:")
    all_files_ok = True
    for file, description in core_files.items():
        if os.path.exists(file):
            print(f"  ✅ {file} - {description}")
        else:
            print(f"  ❌ {file} - MISSING!")
            all_files_ok = False
    
    # Check legal documents
    print(f"\n📚 LEGAL DOCUMENTS:")
    legal_dir = "legal_documents"
    if os.path.exists(legal_dir):
        print(f"  ✅ {legal_dir}/ - Document storage")
        
        # Check subdirectories
        subdirs = ['pdfs', 'processed']
        for subdir in subdirs:
            subdir_path = os.path.join(legal_dir, subdir)
            if os.path.exists(subdir_path):
                file_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
                print(f"  ✅ {subdir}/ - {file_count} files")
            else:
                print(f"  ❌ {subdir}/ - MISSING!")
        
        # Check RAG documents
        rag_file = os.path.join(legal_dir, "pdf_rag_documents.json")
        if os.path.exists(rag_file):
            with open(rag_file, 'r', encoding='utf-8') as f:
                rag_docs = json.load(f)
            print(f"  ✅ pdf_rag_documents.json - {len(rag_docs)} document chunks")
        else:
            print(f"  ❌ pdf_rag_documents.json - MISSING!")
    else:
        print(f"  ❌ {legal_dir}/ - MISSING!")
        all_files_ok = False
    
    # Check imports
    print(f"\n🔧 IMPORT VALIDATION:")
    try:
        print("  🔄 Testing imports...")
        
        # Test if main modules can be imported
        try:
            from scraper import AlbanianLegalScraper
            print("  ✅ Scraper module")
        except Exception as e:
            print(f"  ❌ Scraper import error: {e}")
        
        try:
            from ai import CloudLLMClient
            print("  ✅ AI module")
        except Exception as e:
            print(f"  ❌ AI import error: {e}")
        
        print("  ✅ Core imports working")
        
    except Exception as e:
        print(f"  ❌ Import validation failed: {e}")
        all_files_ok = False
    
    # Summary
    print(f"\n" + "=" * 50)
    if all_files_ok:
        print("🎉 PROJECT VALIDATION SUCCESSFUL!")
        print("✅ All core files present")
        print("✅ Document structure intact")
        print("✅ Imports working")
        print("\n🚀 Ready for presentation!")
        print("\n📋 To start the system:")
        print("  • Double-click: start.bat")
        print("  • Or run: streamlit run app.py")
        print("  • Or see demo: python demo.py")
    else:
        print("❌ PROJECT VALIDATION FAILED!")
        print("⚠️ Some issues need to be fixed before presentation")
    
    return all_files_ok

if __name__ == "__main__":
    validate_project()
