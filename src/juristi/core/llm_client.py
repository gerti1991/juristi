"""
Cloud LLM Integration for Albanian Legal RAG System
Support for Groq API, Google Gemini, and other providers
"""

import os
import requests
import json
from typing import List, Dict, Optional
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

class CloudLLMClient:
    """Client for cloud-based LLM services supporting multiple providers"""
    
    def __init__(self, provider="gemini"):
        """Initialize cloud LLM client with environment variables"""
        self.provider = provider
        self.current_provider = provider  # Add missing attribute
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()
        self.model = self._get_model()
        
        # Initialize Gemini client if needed
        if self.provider == "gemini" and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
        
    def _get_api_key(self) -> str:
        """Get API key from environment or config"""
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                # Check config.py
                try:
                    import config
                    api_key = getattr(config, 'GEMINI_API_KEY', None)
                except ImportError:
                    pass
                
                if not api_key:
                    print("⚠️ GEMINI_API_KEY not found! Get your free API key from: https://aistudio.google.com/app/apikey")
                    return ""
            return api_key
        elif self.provider == "groq":
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
        if self.provider == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta/models"
        elif self.provider == "groq":
            return "https://api.groq.com/openai/v1/chat/completions"
        elif self.provider == "huggingface":
            return "https://api-inference.huggingface.co/models/"
        else:
            return ""
    
    def _get_model(self) -> str:
        """Get model name for the provider"""
        if self.provider == "gemini":
            return "gemini-2.0-flash-exp"  # Free and powerful model
        elif self.provider == "groq":
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
            
            if self.provider == "gemini":
                return self._gemini_request(prompt)
            elif self.provider == "groq":
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
        """Create a comprehensive legal analysis prompt that synthesizes multiple documents"""
        
        # Extract key information from documents
        legal_sources = []
        all_articles = []
        
        for doc in relevant_docs[:5]:  # Use top 5 most relevant documents
            title = doc.get('title', 'Dokument pa titull')
            content = doc.get('content', '')
            source = doc.get('source', 'Pa burim')
            
            # Extract law codes and articles
            import re
            articles = re.findall(r'[Nn]eni\s*\d+[a-z]*', content)
            all_articles.extend(articles)
            
            legal_sources.append({
                'title': title,
                'source': source,
                'content': content[:800],  # Limit content length but keep substantial info
                'articles': articles[:3] if articles else []
            })
        
        # Build comprehensive synthesis prompt
        sources_text = ""
        for i, source in enumerate(legal_sources, 1):
            sources_text += f"""
DOKUMENTI {i}: {source['title']}
Burimi: {source['source']}
Nenet e përfshira: {', '.join(source['articles']) if source['articles'] else 'Të ndryshme'}

Përmbajtja:
{source['content']}

---"""
        
        prompt = f"""Ti je një ekspert i lartë i së drejtës shqiptare me dekada përvojë në interpretimin e legjislacionit shqiptar. Të jepet një pyetje ligjore dhe disa dokumente relevante.

PYETJA E KLIENTIT: {query}

DOKUMENTET LIGJORE PËR ANALIZË:
{sources_text}

DETYRA JOTE:
Krijo një analizë të plotë juridike që SINTETIZON të gjitha dokumentet e mësipërme për të dhënë një përgjigje të vetme, koherente dhe autoritare. 

STRUKTURA E PËRGJIGJES:

🎯 **PËRGJIGJJA E DREJTPËRDREJTË**
Jep një përgjigje të qartë dhe konkrete për pyetjen e bërë.

⚖️ **BAZAT LIGJORE**
Lista të gjitha nenet, kodet dhe ligjet relevante nga dokumentet e analizuara.

📖 **ANALIZA JURIDIKE** 
Shpjego logjikën ligjore duke kombinuar informacionin nga të gjitha dokumentet. Trego se si ligjet e ndryshme lidhen dhe ndikojnë tek njëra-tjetra.

⚠️ **RASTE TË VEÇANTA & PËRJASHTIME**
Nëse ka përjashtime, kushte të veçanta ose nuanca të rëndësishme.

🏛️ **KËSHILLA PRAKTIKE**
Sugjerime konkrete për personin që pyet.

📋 **HAPA TË MËTEJSHËM**
Çfarë duhet të bëjë personi në vazhdim.

RREGULLA TË RËNDËSISHME:
- Përdor VETËM informacionin nga dokumentet e dhëna
- SINTETIZO dokumentet për të krijuar një përgjigje të vetme koherente
- MOS listosh dokumente individualisht - kombinoji ato
- Përdor gjuhën shqipe profesionale dhe të qartë
- Jep një përgjigje autoritare dhe të sigurt
- Nëse ka informacion kontraditor, shpjegoje qartë

FORMATI: Përdor emoji dhe formatim të qartë si në strukturën e mësipërme."""
        
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
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            error_msg = f"Groq API error: {response.status_code}"
            print(error_msg)
            return f"Na vjen keq, ndodhi një gabim: {error_msg}"
    
    def _gemini_request(self, prompt: str) -> str:
        """Make request to Google Gemini API"""
        if not self.gemini_model:
            return self._fallback_response("", [])
            
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1000,
                )
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return f"Na vjen keq, ndodhi një gabim me Gemini API: {e}"
    
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
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'Nuk u gjend përgjigje.')
            return 'Përgjigje e pavlefshme nga API.'
        else:
            return f"Gabim API: {response.status_code}"
    
    def _fallback_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Provide template-based fallback response"""
        if not relevant_docs:
            return """
🏛️ **Sistemi Ligjor Shqiptar - Informacion i Përgjithshëm**

Na vjen keq, nuk gjeta dokumente specifike për pyetjen tuaj. Ja disa sugjerime:

📚 **Për informacion të detajuar ligjor:**
- Vizitoni faqen zyrtare: qbz.gov.al
- Kontaktoni një jurist të kualifikuar
- Konsultohuni me Dhomën e Avokatisë së Shqipërisë

⚖️ **Sistemi ligjor shqiptar bazohet në:**
- Kushtetutën e Republikës së Shqipërisë
- Kodet e ndryshme (Penal, Civil, Pune, etj.)
- Ligjet e veçanta të miratuara nga Kuvendi

**Kujdes:** Ky është vetëm një sistem i përgjithshëm këshillimi. Për çështje të rëndësishme ligjore, rekomandohet konsultimi me ekspertë ligjorë.
            """
        
        # If we have documents, provide a basic summary
        response = "🏛️ **Informacion nga Dokumentet Ligjore**\n\n"
        
        for i, doc in enumerate(relevant_docs[:2], 1):
            response += f"**{i}. {doc['title']}**\n"
            response += f"{doc['content'][:300]}...\n\n"
        
        response += "\n⚖️ **Rekomandim:** Për interpretim të saktë ligjor, konsultohuni me një jurist të kualifikuar."
        
        return response


# Helper functions for backward compatibility
def generate_enhanced_response(query: str, relevant_docs: List[Dict], context: str = "") -> str:
    """Legacy function for backward compatibility"""
    client = CloudLLMClient("gemini")
    return client.generate_response(query, relevant_docs, context)
