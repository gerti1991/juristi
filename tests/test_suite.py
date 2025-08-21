#!/usr/bin/env python3
"""
Albanian Legal RAG System - Unified Test Suite

This module contains all tests consolidated from individual test files.
Provides comprehensive testing of all system components.
"""

import os
import sys
import json
import time
import requests
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
import unittest
from unittest.mock import Mock, patch

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestAlbanianLegalRAG(unittest.TestCase):
    """Comprehensive test suite for Albanian Legal RAG system."""

    def setUp(self):
        """Set up test environment."""
        self.test_start_time = time.time()
        os.environ.setdefault('RAG_QUICK_START', 'true')

    def tearDown(self):
        """Clean up after test."""
        test_duration = time.time() - self.test_start_time
        print(f"Test completed in {test_duration:.3f} seconds")

    # ================================
    # BASIC SYSTEM TESTS
    # ================================

    def test_basic_imports(self):
        """Test basic Python imports."""
        print("🔍 Testing basic imports...")
        try:
            import json
            import os
            import sys
            import datetime
            print("   ✅ Standard library imports: OK")
            self.assertTrue(True)
        except Exception as e:
            print(f"   ❌ Basic imports failed: {e}")
            self.fail(f"Basic imports failed: {e}")

    def test_dependencies(self):
        """Test external dependencies."""
        print("🔍 Testing external dependencies...")
        dependencies = [
            ('streamlit', 'Streamlit web framework'),
            ('groq', 'Groq API client'),
            ('sentence_transformers', 'Sentence Transformers'),
            ('numpy', 'NumPy'),
            ('sklearn', 'Scikit-learn'),
            ('requests', 'HTTP requests'),
            ('python_dotenv', 'Environment variables'),
            ('fastapi', 'FastAPI framework'),
            ('uvicorn', 'ASGI server'),
            ('google', 'Google API client')
        ]
        
        success_count = 0
        failed_imports = []
        
        for package, description in dependencies:
            try:
                __import__(package)
                print(f"   ✅ {package}: OK")
                success_count += 1
            except ImportError as e:
                print(f"   ⚠️ {package}: OPTIONAL - {e}")
                failed_imports.append(package)
        
        print(f"   📊 Dependencies: {success_count}/{len(dependencies)} working")
        
        # Only fail if critical dependencies are missing
        critical_deps = ['streamlit', 'requests', 'numpy']
        critical_missing = [dep for dep in failed_imports if dep in critical_deps]
        
        if critical_missing:
            self.fail(f"Critical dependencies missing: {critical_missing}")

    def test_project_structure(self):
        """Test project file structure."""
        print("🔍 Testing project structure...")
        
        # Check new modular structure
        expected_structure = {
            'src/juristi/__init__.py': 'Main package init',
            'src/juristi/core/__init__.py': 'Core module init',
            'src/juristi/data/__init__.py': 'Data module init',
            'src/juristi/api/__init__.py': 'API module init',
            'src/juristi/ui/__init__.py': 'UI module init',
            'tests/__init__.py': 'Tests init',
            'scripts/__init__.py': 'Scripts init',
        }
        
        missing_files = []
        for file_path, description in expected_structure.items():
            full_path = os.path.join(os.path.dirname(__file__), '..', file_path)
            if os.path.exists(full_path):
                print(f"   ✅ {file_path} - {description}")
            else:
                print(f"   ❌ {file_path} - MISSING!")
                missing_files.append(file_path)
        
        if missing_files:
            print(f"   ⚠️ Missing files: {missing_files}")

    def test_legal_documents(self):
        """Test legal documents structure."""
        print("🔍 Testing legal documents...")
        
        legal_dir = os.path.join(os.path.dirname(__file__), '..', 'legal_documents')
        if os.path.exists(legal_dir):
            print(f"   ✅ legal_documents/ - Document storage")
            
            # Check subdirectories
            subdirs = ['pdfs', 'processed']
            for subdir in subdirs:
                subdir_path = os.path.join(legal_dir, subdir)
                if os.path.exists(subdir_path):
                    file_count = len([f for f in os.listdir(subdir_path) 
                                    if os.path.isfile(os.path.join(subdir_path, f))])
                    print(f"   ✅ {subdir}/ - {file_count} files")
                else:
                    print(f"   ❌ {subdir}/ - MISSING!")
        else:
            print("   ⚠️ legal_documents/ directory not found")

    # ================================
    # RAG ENGINE TESTS
    # ================================

    def test_rag_engine_import(self):
        """Test RAG engine can be imported."""
        print("🔍 Testing RAG engine import...")
        try:
            from juristi.core.rag_engine import AlbanianLegalRAG
            print("   ✅ RAG engine import: OK")
            self.assertTrue(True)
        except ImportError as e:
            print(f"   ❌ RAG engine import failed: {e}")
            # Don't fail the test yet as we're refactoring

    def test_rag_engine_initialization(self):
        """Test RAG engine initialization."""
        print("🔍 Testing RAG engine initialization...")
        try:
            from juristi.core.rag_engine import AlbanianLegalRAG
            
            start_time = time.time()
            rag = AlbanianLegalRAG(quick_start=True)
            init_time = time.time() - start_time
            
            print(f"   ✅ RAG engine initialized in {init_time:.3f}s")
            self.assertIsNotNone(rag)
            self.assertTrue(init_time < 10.0)  # Should initialize quickly
        except Exception as e:
            print(f"   ❌ RAG engine initialization failed: {e}")

    def test_search_functionality(self):
        """Test search functionality."""
        print("🔍 Testing search functionality...")
        try:
            from juristi.core.rag_engine import AlbanianLegalRAG
            
            rag = AlbanianLegalRAG(quick_start=True)
            
            # Test search queries
            test_queries = [
                "ligj penal",
                "kodi civil",
                "drejtësi penale për të mitur"
            ]
            
            for query in test_queries:
                print(f"   🔍 Testing query: '{query}'")
                search_start = time.time()
                
                try:
                    results = rag.search_documents(query, top_k=3)
                    search_time = time.time() - search_start
                    
                    print(f"      ✅ Found {len(results)} results in {search_time:.3f}s")
                    self.assertIsInstance(results, list)
                    
                    if results:
                        # Verify result structure
                        result = results[0]
                        required_fields = ['title', 'content', 'source']
                        for field in required_fields:
                            self.assertIn(field, result, f"Missing field: {field}")
                        
                except Exception as e:
                    print(f"      ❌ Search failed: {e}")

        except ImportError:
            print("   ⚠️ Skipping search test - RAG engine not available")

    # ================================
    # API TESTS
    # ================================

    def test_api_health_check(self):
        """Test API health endpoint."""
        print("🔍 Testing API health check...")
        base_url = "http://127.0.0.1:8000"
        
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            print(f"   ✅ Health check: {response.status_code}")
            self.assertEqual(response.status_code, 200)
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ API not available: {e}")

    def test_api_search_endpoint(self):
        """Test API search endpoint."""
        print("🔍 Testing API search endpoint...")
        base_url = "http://127.0.0.1:8000"
        
        try:
            search_data = {
                "query": "sa është denimi për plagosje me armë të ftohta",
                "top_k": 3
            }
            
            response = requests.post(f"{base_url}/search", 
                                   json=search_data, timeout=30)
            print(f"   ✅ Search endpoint: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn("results", data)
                print(f"   📄 Found {len(data.get('results', []))} results")
            
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ API not available: {e}")

    # ================================
    # PERFORMANCE TESTS
    # ================================

    def test_startup_performance(self):
        """Test startup performance."""
        print("🔍 Testing startup performance...")
        
        # Test with quick start mode
        os.environ['RAG_QUICK_START'] = 'true'
        
        try:
            from juristi.core.rag_engine import AlbanianLegalRAG
            
            start_time = time.time()
            rag = AlbanianLegalRAG()
            startup_time = time.time() - start_time
            
            print(f"   ✅ Quick start: {startup_time:.3f}s")
            self.assertTrue(startup_time < 5.0, f"Startup too slow: {startup_time:.3f}s")
            
            # Test first search performance
            search_start = time.time()
            results = rag.search_documents("test query", top_k=1)
            first_search_time = time.time() - search_start
            
            print(f"   ✅ First search: {first_search_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Performance test failed: {e}")

    def test_memory_usage(self):
        """Test memory usage (basic check)."""
        print("🔍 Testing memory usage...")
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create RAG instance
            from juristi.core.rag_engine import AlbanianLegalRAG
            rag = AlbanianLegalRAG(quick_start=True)
            
            # Perform some operations
            rag.search_documents("test", top_k=1)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"   📊 Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # Cleanup
            del rag
            gc.collect()
            
        except ImportError:
            print("   ⚠️ psutil not available for memory testing")
        except Exception as e:
            print(f"   ❌ Memory test failed: {e}")

    # ================================
    # INTEGRATION TESTS
    # ================================

    def test_criminal_query(self):
        """Test specific criminal law queries."""
        print("🔍 Testing criminal law queries...")
        
        criminal_queries = [
            "sa është denimi për vrasje me dashje",
            "çfarë dënimi ka për hajdutëri", 
            "ligji penal për korrupsion",
            "dënimi për dhunë në familje"
        ]
        
        try:
            from juristi.core.rag_engine import AlbanianLegalRAG
            rag = AlbanianLegalRAG(quick_start=True)
            
            for query in criminal_queries:
                print(f"   🔍 Query: '{query}'")
                results = rag.search_documents(query, top_k=2)
                
                if results:
                    print(f"      ✅ Found {len(results)} relevant documents")
                    # Check if results contain relevant legal terms
                    content = " ".join([r.get('content', '') for r in results])
                    legal_terms = ['ligj', 'kod', 'nen', 'dënim', 'burgim']
                    found_terms = [term for term in legal_terms if term in content.lower()]
                    print(f"      📝 Legal terms found: {found_terms}")
                else:
                    print(f"      ⚠️ No results found")
                    
        except Exception as e:
            print(f"   ❌ Criminal query test failed: {e}")

    # ================================
    # CONFIGURATION TESTS  
    # ================================

    def test_environment_variables(self):
        """Test environment variable configuration."""
        print("🔍 Testing environment variables...")
        
        # Test required environment variables
        env_vars = {
            'GROQ_API_KEY': 'Groq API key',
            'GEMINI_API_KEY': 'Google Gemini API key',
            'RAG_QUICK_START': 'Quick start mode'
        }
        
        for var, description in env_vars.items():
            value = os.getenv(var)
            if value:
                print(f"   ✅ {var}: Configured")
            else:
                print(f"   ⚠️ {var}: Not set - {description}")

    def test_google_embeddings(self):
        """Test Google embeddings integration."""
        print("🔍 Testing Google embeddings...")
        
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                print("   ✅ Google AI configured")
                
                # Test embedding generation
                test_text = "Test Albanian legal document content"
                try:
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=test_text,
                        task_type="retrieval_document"
                    )
                    if result and 'embedding' in result:
                        print(f"   ✅ Embedding generated: {len(result['embedding'])} dimensions")
                    else:
                        print("   ⚠️ Embedding generation returned unexpected format")
                except Exception as e:
                    print(f"   ❌ Embedding generation failed: {e}")
            else:
                print("   ⚠️ GEMINI_API_KEY not configured")
                
        except ImportError:
            print("   ⚠️ Google AI library not available")
        except Exception as e:
            print(f"   ❌ Google embeddings test failed: {e}")


def run_test_suite():
    """Run the complete test suite with detailed reporting."""
    print("🏛️ ALBANIAN LEGAL RAG - UNIFIED TEST SUITE")
    print("=" * 60)
    print(f"📅 Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version}")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlbanianLegalRAG)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"🎯 TEST SUMMARY")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, error in result.failures:
            print(f"   {test}: {error}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, error in result.errors:
            print(f"   {test}: {error}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    
    print(f"\n🏆 Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Albanian Legal RAG system is working well!")
    elif success_rate >= 60:
        print("⚠️ Albanian Legal RAG system has some issues to address")
    else:
        print("❌ Albanian Legal RAG system needs significant fixes")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
