#!/usr/bin/env python3
"""
Test for the modern Albanian Legal RAG system with multi-provider embeddings
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the Python path
sys.path.append('src')

def test_modern_rag():
    """Test the modern RAG system with dynamic dimension handling"""
    print("🧪 Testing Modern Albanian Legal RAG System")
    print("=" * 50)
    
    # Check if we have the Google API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No Google API key found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")
        return False
    
    try:
        # Import the modern system
        from juristi.core.rag_engine import AlbanianLegalRAG
        print("✅ Successfully imported AlbanianLegalRAG")
        
        # Initialize the system
        print("🔄 Initializing modern RAG system...")
        rag_system = AlbanianLegalRAG(verbose=True)
        print(f"✅ System initialized with {rag_system.active_embedding_provider} embeddings")
        
        # Load documents with dynamic dimension handling
        print("📚 Loading documents...")
        success = rag_system.load_documents_from_directory("legal_documents")
        
        if success:
            print("✅ Documents loaded successfully!")
            
            # Get system status
            status = rag_system.get_system_status()
            print(f"📊 Total documents: {status.get('total_documents', 0)}")
            print(f"🎯 Active embedding provider: {rag_system.active_embedding_provider}")
            
            # Test queries
            test_queries = [
                "Çfarë është kodi civil?",
                "Si funksionon sistemi gjyqësor në Shqipëri?",
                "Të drejtat e punëtorëve?"
            ]
            
            print("\n🔍 Testing queries...")
            for i, query in enumerate(test_queries, 1):
                print(f"\nQuery {i}: {query}")
                try:
                    result = rag_system.query(query)
                    if result:
                        answer = result.get('answer', str(result)) if isinstance(result, dict) else str(result)
                        print(f"✅ Success - Answer length: {len(answer)} chars")
                        print(f"📄 Preview: {answer[:150]}...")
                    else:
                        print("⚠️ Empty response")
                except Exception as e:
                    print(f"❌ Query failed: {e}")
            
            return True
        else:
            print("❌ Failed to load documents")
            return False
        
        # Test query if documents loaded
        test_query = "Çfarë thotë ligji për martesën?"
        print(f"\n📝 Testing query: '{test_query}'")
        
        try:
            result = rag_engine.query(test_query)
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
                return False
            else:
                print(f"✅ Answer: {result['answer'][:200]}...")
                print(f"📊 Found {len(result['sources'])} sources")
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            return False
        
        print("\n🎉 Modern RAG system test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install langchain-google-genai langchain-community chromadb")
        return False
        
    except Exception as e:
        print(f"❌ Error testing modern RAG: {e}")
        return False

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = test_modern_rag()
    if not success:
        sys.exit(1)
