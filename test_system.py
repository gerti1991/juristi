#!/usr/bin/env python3
"""
Albanian Legal RAG System - Comprehensive Test Suite
Tests all major components and functionality
"""

import os
import sys
import json
import traceback
from datetime import datetime

def test_basic_imports():
    """Test basic Python imports"""
    print("🔍 Testing basic imports...")
    try:
        import json
        import os
        import sys
        print("   ✅ Standard library imports: OK")
        return True
    except Exception as e:
        print(f"   ❌ Basic imports failed: {e}")
        return False

def test_dependencies():
    """Test external dependencies"""
    print("🔍 Testing external dependencies...")
    dependencies = [
        ('streamlit', 'Streamlit web framework'),
        ('groq', 'Groq API client'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('requests', 'HTTP requests'),
        ('dotenv', 'Environment variables')
    ]
    
    success_count = 0
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"   ✅ {package}: OK")
            success_count += 1
        except ImportError as e:
            print(f"   ❌ {package}: FAILED - {e}")
    
    print(f"   📊 Dependencies: {success_count}/{len(dependencies)} working")
    return success_count == len(dependencies)

def test_project_structure():
    """Test project file structure"""
    print("🔍 Testing project structure...")
    
    required_files = [
        'app.py',
        'ai.py', 
        'scraper.py',
        'config.py',
        'requirements.txt',
        '.env',
        'README.md'
    ]
    
    required_dirs = [
        'legal_documents',
        'legal_documents/pdfs',
        'legal_documents/processed'
    ]
    
    success_count = 0
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}: EXISTS")
            success_count += 1
        else:
            print(f"   ❌ {file}: MISSING")
    
    # Check directories  
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}/: EXISTS")
            success_count += 1
        else:
            print(f"   ❌ {directory}/: MISSING")
    
    total_items = len(required_files) + len(required_dirs)
    print(f"   📊 Project structure: {success_count}/{total_items} items found")
    return success_count >= total_items * 0.8  # 80% threshold

def test_legal_documents():
    """Test legal document loading"""
    print("🔍 Testing legal documents...")
    
    try:
        pdf_file = os.path.join("legal_documents", "pdf_rag_documents.json")
        
        if not os.path.exists(pdf_file):
            print("   ❌ PDF documents file not found")
            return False
            
        with open(pdf_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        if len(docs) == 0:
            print("   ❌ No documents loaded")
            return False
            
        print(f"   ✅ {len(docs)} document chunks loaded")
        
        # Count document types
        doc_types = {}
        for doc in docs:
            title = doc.get('title', 'Unknown')
            doc_types[title] = doc_types.get(title, 0) + 1
        
        print(f"   ✅ {len(doc_types)} different legal codes found")
        
        # Check key documents
        key_docs = [
            "Kodi Civil 2023",
            "Kodi I Punes Ligj 2024",
            "Ligj Nr. 7895 1995 Kodi Penal I Përditësuar"
        ]
        
        found_key_docs = 0
        for key_doc in key_docs:
            if any(key_doc in title for title in doc_types.keys()):
                found_key_docs += 1
                
        print(f"   ✅ {found_key_docs}/{len(key_docs)} key legal codes found")
        return True
        
    except Exception as e:
        print(f"   ❌ Legal documents test failed: {e}")
        return False

def test_environment_config():
    """Test environment configuration"""
    print("🔍 Testing environment configuration...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check API key
        api_key = os.getenv('GROQ_API_KEY')
        if api_key and len(api_key) > 10:
            print("   ✅ Groq API key configured")
        else:
            print("   ⚠️  Groq API key not found or invalid")
            
        # Check other config
        model = os.getenv('GROQ_MODEL', 'not set')
        print(f"   ✅ Groq model: {model}")
        
        threshold = os.getenv('SIMILARITY_THRESHOLD', 'not set')
        print(f"   ✅ Similarity threshold: {threshold}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Environment config test failed: {e}")
        return False

def test_core_modules():
    """Test core application modules"""
    print("🔍 Testing core modules...")
    
    modules_to_test = [
        ('ai', 'CloudLLMClient'),
        ('scraper', 'AlbanianLegalScraper'),
        ('config', None)
    ]
    
    success_count = 0
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            if class_name:
                getattr(module, class_name)
                print(f"   ✅ {module_name}.{class_name}: OK")
            else:
                print(f"   ✅ {module_name}: OK")
            success_count += 1
        except Exception as e:
            print(f"   ❌ {module_name}: FAILED - {e}")
    
    print(f"   📊 Core modules: {success_count}/{len(modules_to_test)} working")
    return success_count == len(modules_to_test)

def test_rag_functionality():
    """Test RAG system functionality"""
    print("🔍 Testing RAG functionality...")
    
    try:
        # Test sentence transformer
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test embedding
        test_text = "Sa dit pushimi kam në punë?"
        embedding = model.encode([test_text])
        
        if embedding is not None and len(embedding) > 0:
            print("   ✅ Sentence embedding: OK")
        else:
            print("   ❌ Sentence embedding: FAILED")
            return False
            
        # Test similarity calculation
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        embedding1 = model.encode(["test text 1"])
        embedding2 = model.encode(["test text 2"])
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        print(f"   ✅ Similarity calculation: OK (score: {similarity:.3f})")
        return True
        
    except Exception as e:
        print(f"   ❌ RAG functionality test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🧪 ALBANIAN LEGAL RAG SYSTEM - FULL TEST SUITE")
    print("=" * 60)
    print(f"🕒 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Dependencies", test_dependencies),
        ("Project Structure", test_project_structure),
        ("Legal Documents", test_legal_documents),
        ("Environment Config", test_environment_config),
        ("Core Modules", test_core_modules),
        ("RAG Functionality", test_rag_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"   🎉 {test_name}: PASSED")
            else:
                print(f"   💥 {test_name}: FAILED")
        except Exception as e:
            print(f"   💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 TEST SUMMARY")
    print("-" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print()
    print(f"📈 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your Albanian Legal RAG system is ready!")
        print("🚀 You can now run: streamlit run app.py")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. System should work with minor issues.")
    else:
        print("❌ Several tests failed. Please check the issues above.")
    
    print(f"🕒 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return passed == total

if __name__ == "__main__":
    run_all_tests()
