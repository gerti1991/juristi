#!/usr/bin/env python3
"""
Quick test of the enhanced Albanian Legal RAG System with Google embeddings
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_system_quick():
    """Quick test of the enhanced system"""
    
    print("🧪 QUICK SYSTEM TEST - Google Embeddings")
    print("=" * 60)
    
    try:
        # Import and initialize
        print("🔄 Importing system...")
        from app import CloudEnhancedAlbanianLegalRAG
        
        print("🔄 Initializing RAG system...")
        rag_system = CloudEnhancedAlbanianLegalRAG()
        
        # Check status
        print(f"📊 Google embeddings enabled: {rag_system.use_google_embeddings}")
        print(f"📊 Total documents: {len(rag_system.legal_documents)}")
        print(f"📊 Embeddings ready: {rag_system.document_embeddings is not None}")
        
        if rag_system.document_embeddings is not None:
            print(f"📊 Embedding shape: {rag_system.document_embeddings.shape}")
        
        # Test the criminal law query
        print("\n🔍 Testing criminal law query...")
        test_query = "sa eshte denimi per plagosje me arme te ftohte"
        print(f"Query: {test_query}")
        
        # Test search
        relevant_docs = rag_system.search_documents(test_query, top_k=3)
        print(f"\n📋 Found {len(relevant_docs)} relevant documents:")
        
        for i, doc in enumerate(relevant_docs, 1):
            similarity = doc.get('similarity', 0)
            title = doc.get('title', 'Unknown')[:60]
            print(f"  {i}. {title}... (Score: {similarity:.3f})")
        
        # Test full response
        print("\n🤖 Testing full response generation...")
        response = rag_system.search_legal_documents(test_query)
        
        print(f"\n📝 Response length: {len(response)} characters")
        print("📋 Response preview:")
        print("-" * 40)
        print(response[:500] + "..." if len(response) > 500 else response)
        print("-" * 40)
        
        print("\n✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system_quick()
    exit(0 if success else 1)
