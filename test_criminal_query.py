#!/usr/bin/env python3
"""
Test specific criminal law query for Albanian Legal RAG System
"""

import os
import sys
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import CloudEnhancedAlbanianLegalRAG

def test_criminal_query():
    """Test the specific criminal law query about assault with cold weapons"""
    
    print("🧪 TESTING CRIMINAL LAW QUERY")
    print("=" * 50)
    
    try:
        # Initialize the system
        print("🔄 Loading system...")
        rag_system = CloudEnhancedAlbanianLegalRAG()
        
        # Test query enhancement first
        test_query = "sa eshte denimi per plagosje me arme te ftohte"
        enhanced_query = rag_system._enhance_albanian_query(test_query)
        
        print(f"📝 Original Query: {test_query}")
        print(f"✨ Enhanced Query: {enhanced_query}")
        print()
        
        # Test document search
        print("🔍 Searching for relevant documents...")
        similar_docs = rag_system._find_similar_documents(enhanced_query)
        
        print(f"📊 Found {len(similar_docs)} relevant documents")
        
        if similar_docs:
            print("\n📋 Top relevant documents:")
            for i, (doc, score) in enumerate(similar_docs[:5], 1):
                print(f"   {i}. Score: {score:.3f} - {doc['title'][:100]}...")
        else:
            print("⚠️ No relevant documents found!")
        
        # Test full response
        print("\n🤖 Getting AI response...")
        response = rag_system.search_legal_documents(test_query)
        
        print("\n📋 SYSTEM RESPONSE:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_criminal_query()
    exit(0 if success else 1)
