#!/usr/bin/env python3
"""
Test script for the improved Albanian Legal RAG system
"""

import sys
import os
sys.path.append('src')

from juristi.core.rag_engine import AlbanianLegalRAG
from juristi.core.llm_client import CloudLLMClient

def test_improved_system():
    print("🧪 Testing improved Albanian Legal RAG system...")
    
    # Initialize system with optimized settings
    print("🔄 Initializing RAG system...")
    rag_system = AlbanianLegalRAG(
        quick_start=False,  # Full initialization for testing
        rag_mode='traditional',
        use_google_embeddings=True
    )
    
    # Test queries
    test_queries = [
        "sa vite denohesh per plagosje",
        "cfare eshte martesa sipas kodit civil",
        "procedura e divorcit ne shqiperi"
    ]
    
    print(f"\n📋 Testing with {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"🔍 Query {i}: {query}")
        print(f"{'='*50}")
        
        # Search for documents
        results = rag_system.search_documents(query, top_k=3)
        print(f"📚 Found {len(results)} documents")
        
        if results:
            # Initialize LLM client
            llm_client = CloudLLMClient("gemini")
            
            # Generate AI response
            print("🤖 Generating AI response...")
            ai_response = llm_client.generate_response(query, results)
            
            print("🎯 AI Response:")
            print("-" * 40)
            print(ai_response)
            print("-" * 40)
        else:
            print("❌ No documents found")
        
        print(f"⏱️ Query {i} completed\n")
    
    print("✅ Test completed!")

if __name__ == "__main__":
    test_improved_system()
