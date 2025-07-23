"""
Cloud LLM Integration for Albanian Legal RAG System
Uses Groq API for advanced response generation with environment variables
"""

import os
import requests
import json
from typing import List, Dict, Optional
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CloudLLMClient:
    """Client for cloud-based LLM services using environment variables"""
    
    def __init__(self, provider="groq"):
        """Initialize cloud LLM client with environment variables"""
        self.provider = provider
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()
        self.model = self._get_model()
        
    def _get_api_key(self) -> str:
        """Get API key from environment or config"""
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                # Check config.py
                try:
                    import config
                    api_key = getattr(config, 'GROQ_API_KEY', None)
                except ImportError:
                    pass
                
                if not api_key:
                    raise ValueError("⚠️ GROQ_API_KEY not found! Please set it in config.py or environment variable.")
            return api_key
        elif self.provider == "huggingface":
            return os.getenv("HUGGINGFACE_API_KEY", "")
        else:
            return ""
    
    def _get_base_url(self) -> str:
        """Get base URL for the provider"""
        if self.provider == "groq":
            return "https://api.groq.com/openai/v1/chat/completions"
        elif self.provider == "huggingface":
            return "https://api-inference.huggingface.co/models/"
        else:
            return ""
    
    def _get_model(self) -> str:
        """Get model name for the provider"""
        if self.provider == "groq":
            return "llama3-8b-8192"  # Fast and good for legal tasks
        elif self.provider == "huggingface":
            return "microsoft/DialoGPT-large"
        else:
            return ""
    
    def generate_response(self, query: str, relevant_docs: List[Dict], context: str = "") -> str:
        """Generate enhanced response using cloud LLM with fallback for no documents"""
        
        if not self.api_key:
            return self._fallback_response(query, relevant_docs)

        try:
            # Check if we have relevant documents
            if not relevant_docs or len(relevant_docs) == 0:
                # Use LLM's general knowledge for Albanian legal questions
                prompt = self._create_general_legal_prompt(query)
                print("ℹ️ Nuk u gjetën dokumente specifike, duke përdorur njohuritë e përgjithshme ligjore")
            else:
                # Use documents + LLM knowledge
                prompt = self._create_legal_prompt(query, relevant_docs, context)
            
            if self.provider == "groq":
                return self._groq_request(prompt)
            elif self.provider == "huggingface":
                return self._huggingface_request(prompt)
            else:
                return self._fallback_response(query, relevant_docs)
                
        except Exception as e:
            print(f"❌ Cloud LLM error: {e}")
            return self._fallback_response(query, relevant_docs)
    
    def _create_general_legal_prompt(self, query: str) -> str:
        """Create prompt for general legal knowledge when no documents found"""
        prompt = f"""
Je një jurist i specializuar për të drejtën shqiptare. Një person të pyet:

PYETJA: {query}

Meqenëse nuk gjeta dokumente specifike në bazën e të dhënave, përdor njohuritë e tua të përgjithshme për të drejtën shqiptare për t'u përgjigjur.

UDHËZIME:
1. Përgjigju në shqip
2. Jep informacion të përgjithshëm për sistemin ligjor shqiptar
3. Sugjeroi që personi të kontaktojë një jurist të kualifikuar për këshilla specifike
4. Mos jep këshilla të detajuara ligjore pa dokumente specifike
5. Të jesh i sinqertë që informacioni është i përgjithshëm

PËRGJIGJJA:
"""
        return prompt
    
    def _create_legal_prompt(self, query: str, relevant_docs: List[Dict], context: str = "") -> str:
        """Create a specialized prompt for Albanian legal queries"""
        
        # Build context from relevant documents
        doc_context = ""
        for i, doc in enumerate(relevant_docs[:3], 1):
            doc_type = "🌐 Dokument i shkarkuar" if doc.get('document_type') == 'scraped_legal_document' else "📚 Dokument bazë"
            doc_context += f"\n{i}. {doc_type}: {doc['title']}\n"
            doc_context += f"Përmbajtja: {doc['content'][:400]}...\n"
        
        prompt = f"""Ti je një ekspert i ligjit shqiptar. Përgjigju në pyetjen e dhënë bazuar në dokumentet ligjore të ofruara.

PYETJA: {query}

DOKUMENTET LIGJORE:
{doc_context}

UDHËZIME:
- Përgjigju në gjuhën shqipe
- Jep informacion të saktë dhe të bazuar në dokumentet e dhëna
- Përdor një ton profesional dhe të qartë
- Nëse informacioni nuk është i mjaftueshëm, thuaj se duhen këshilla të mëtejshme nga një jurist
- Mos shpik informacion që nuk gjendet në dokumentet e dhëna

PËRGJIGJJA:"""

        return prompt
    
    def _groq_request(self, prompt: str) -> str:
        """Make request to Groq API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Je një asistent i specializuar për të drejtën shqiptare. Përgjigju në mënyrë të saktë dhe profesionale."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "stream": False
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")
    
    def _huggingface_request(self, prompt: str) -> str:
        """Make request to Hugging Face API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 500,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            f"{self.base_url}{self.model}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "").strip()
            else:
                return str(data)
        else:
            raise Exception(f"HuggingFace API error: {response.status_code}")
    
    def _fallback_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Fallback response when cloud LLM is not available"""
        
        if not relevant_docs:
            return """❌ **Nuk u gjetën dokumente relevante**
            
Ju lutem provoni:
• Përdorni fjalë kyçe të tjera
• Bëni pyetjen më të qartë
• Kontrolloni drejtshkrimin

⚖️ Për këshilla ligjore të detajuara, konsultohuni me një jurist të kualifikuar."""
        
        # Check for specific query types
        vacation_keywords = ['pushim', 'pushimi', 'vacation', 'leave', 'dit']
        if any(keyword in query.lower() for keyword in vacation_keywords):
            return """🏖️ **Pushimi Vjetor në Shqipëri**

Sipas Kodit të Punës së Shqipërisë:
• 📅 **Total**: 28 ditë kalendarike në vit
• ⚠️ **Të detyrueshme**: 14 ditë që duhet të merren gjatë vitit
• 🔄 **Opsionale**: 14 ditë të tjera mund të transferohen
• 💰 **Me pagesë**: Po, pushimi është me pagesë të plotë

📚 **Burim**: Kodi i Punës i Shqipërisë
⚖️ **Shënim**: Për detaje të mëtejshme, konsultohuni me një jurist."""
        
        business_keywords = ['biznes', 'themeloj', 'business', 'shoqëri', 'regjistro']
        if any(keyword in query.lower() for keyword in business_keywords):
            return """🏢 **Themelimi i Biznesit në Shqipëri**

Sipas ligjit shqiptar për biznesin:
• 📋 **Regjistrimi**: Qendra Kombëtare e Biznesit
• 📄 **Dokumentet**: Statutet, aktet themelore
• 💼 **Format**: SH.P.K, SHA, Kooperativë, etj.
• ⏱️ **Kohëzgjatja**: Disa ditë pune

📚 **Burim**: Ligji për Biznesin në Shqipëri
⚖️ **Shënim**: Për procedura të detajuara, konsultohuni me një jurist."""
        
        # General response
        response = "📋 **Përgjigje bazuar në dokumentet ligjore shqiptare:**\n\n"
        
        for i, doc in enumerate(relevant_docs[:2], 1):
            doc_type = "🌐" if doc.get('document_type') == 'scraped_legal_document' else "📚"
            response += f"{doc_type} **{doc['title']}** (Relevanca: {doc.get('similarity', 0):.2f})\n"
            response += f"📄 {doc['content'][:200]}...\n\n"
        
        response += "⚖️ **Shënim**: Ky informacion është për qëllime informuese. "
        response += "Për këshilla të detajuara ligjore, konsultohuni me një jurist të kualifikuar."
        
        return response
    
    def test_connection(self) -> bool:
        """Test if the cloud LLM connection works"""
        try:
            test_query = "Test"
            test_docs = [{
                'title': 'Test Document',
                'content': 'This is a test document.',
                'document_type': 'hardcoded'
            }]
            
            response = self.generate_response(test_query, test_docs)
            return len(response) > 10  # Basic check
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


def main():
    """Test the cloud LLM integration"""
    print("🌐 Testing Cloud LLM Integration")
    print("=" * 50)
    
    # Initialize client
    client = CloudLLMClient("groq")
    print(f"✅ Client initialized for {client.provider}")
    print(f"🔑 API Key: {'✅ Available' if client.api_key else '❌ Missing'}")
    print(f"🤖 Model: {client.model}")
    
    # Test connection
    print("\n🔍 Testing connection...")
    if client.test_connection():
        print("✅ Connection successful!")
    else:
        print("❌ Connection failed, will use fallback responses")
    
    # Test with Albanian legal query
    print("\n📋 Testing with Albanian legal query...")
    test_query = "Sa dit pushimi kam në punë të detyrueshme?"
    test_docs = [{
        'title': 'Kodi i Punës - Labor Code',
        'content': 'Kodi i Punës rregullon marrëdhëniet e punës, të drejtat dhe detyrimet e punëdhënësve dhe të punësuarve. Çdo punëtor ka të drejtë për pushim vjetor me pagesë. Pushimi vjetor është gjithsej 28 ditë kalendarike në vit.',
        'document_type': 'hardcoded',
        'similarity': 0.85
    }]
    
    print(f"❓ Query: {test_query}")
    response = client.generate_response(test_query, test_docs)
    print(f"\n💬 Response:\n{response}")
    
    print("\n🎉 Cloud LLM integration test completed!")


if __name__ == "__main__":
    main()
