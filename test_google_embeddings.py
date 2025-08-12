#!/usr/bin/env python3
"""
Test Google embeddings integration for Albanian Legal RAG System
"""

import os
import sys
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def test_google_embeddings():
    """Test Google embeddings API directly"""
    
    print("🧪 TESTING GOOGLE EMBEDDINGS API")
    print("=" * 50)
    
    try:
        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment")
            return False
        
        print(f"✅ API Key found: {api_key[:10]}...")
        
        # Configure Google API
        genai.configure(api_key=api_key)
        print("✅ Google API configured")
        
        # Test with Albanian legal text
        test_texts = [
            "Kodi Penal i Shqipërisë për plagosje me armë të ftohtë",
            "Ligji i Punës për pushimin vjetor të punëtorëve",
            "Kodi Civil për të drejtat e pronësisë"
        ]
        
        print("\n🔍 Testing document embeddings...")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. Testing: {text[:50]}...")
            
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = np.array(result['embedding'])
            print(f"     ✅ Embedding shape: {embedding.shape}")
            print(f"     📊 Sample values: {embedding[:3]}")
        
        # Test query embedding
        print("\n🔍 Testing query embedding...")
        query = "sa eshte denimi per plagosje me arme te ftohte"
        print(f"  Query: {query}")
        
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        
        query_embedding = np.array(result['embedding'])
        print(f"  ✅ Query embedding shape: {query_embedding.shape}")
        print(f"  📊 Sample values: {query_embedding[:3]}")
        
        # Test similarity calculation
        print("\n🔍 Testing similarity calculation...")
        doc_embeddings = []
        for text in test_texts:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            doc_embeddings.append(result['embedding'])
        
        doc_embeddings = np.array(doc_embeddings)
        query_emb = np.array([result['embedding']])
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_emb, doc_embeddings)[0]
        
        print(f"  📊 Similarities: {similarities}")
        
        # Find best match
        best_idx = np.argmax(similarities)
        print(f"  🎯 Best match: Document {best_idx + 1} (score: {similarities[best_idx]:.3f})")
        print(f"     Text: {test_texts[best_idx]}")
        
        print("\n🎉 Google embeddings test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_google_embeddings()
    exit(0 if success else 1)
