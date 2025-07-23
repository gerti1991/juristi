"""
Albanian Legal Document Scraper and Processor
Scrapes legal documents from qbz.gov.al and processes them for the RAG system
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import json
from datetime import datetime

# PDF and document processing
try:
    import PyPDF2
    from docx import Document
except ImportError:
    print("Warning: PDF/DOCX processing libraries not available")

class AlbanianLegalScraper:
    def __init__(self, output_dir='legal_documents', max_docs=20):
        """
        Initialize the legal document scraper
        
        Args:
            output_dir: Directory to store downloaded documents
            max_docs: Maximum number of documents to download per session
        """
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'pdfs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
        
        # Track downloaded documents
        self.metadata_file = os.path.join(output_dir, 'document_metadata.json')
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Load existing document metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {'documents': [], 'last_updated': None}
        return {'documents': [], 'last_updated': None}
    
    def save_metadata(self):
        """Save document metadata"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def clean_filename(self, filename: str) -> str:
        """Clean filename for safe storage"""
        # Remove invalid characters and limit length
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = filename.replace(' ', '_')
        return filename[:100]  # Limit length
    
    def download_pdf(self, url: str, filename: str) -> bool:
        """
        Download a PDF file from URL
        
        Args:
            url: URL to download from
            filename: Local filename to save to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self.session.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                filepath = os.path.join(self.output_dir, 'pdfs', filename)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ Downloaded: {filename}")
                return True
            else:
                print(f"❌ Failed to download {url} (status: {response.status_code})")
                return False
        except Exception as e:
            print(f"❌ Download error for {url}: {e}")
            return False
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"❌ PDF extraction error for {pdf_path}: {e}")
            return ""
    
    def process_text(self, text: str, title: str = "") -> Dict:
        """
        Process extracted text into chunks suitable for RAG
        
        Args:
            text: Raw text content
            title: Document title
            
        Returns:
            Dict: Processed document data
        """
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Split into chunks (approximately 500 characters each)
        chunk_size = 500
        chunks = []
        
        # Try to split by sentences first
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return {
            'title': title,
            'content': text,
            'chunks': chunks,
            'chunk_count': len(chunks),
            'total_length': len(text)
        }
    
    def scrape_qbz_search(self, search_terms: List[str] = None) -> List[Dict]:
        """
        Scrape Albanian legal documents from qbz.gov.al
        
        Args:
            search_terms: List of terms to search for
            
        Returns:
            List[Dict]: List of document metadata
        """
        if search_terms is None:
            search_terms = ['ligj', 'kod', 'vendim', 'dekret']  # law, code, decision, decree
        
        found_documents = []
        
        for term in search_terms:
            print(f"🔍 Searching for: {term}")
            
            try:
                # Search URL for qbz.gov.al
                search_url = f"https://qbz.gov.al/eli/search?query={term}"
                response = self.session.get(search_url, timeout=30)
                
                if response.status_code != 200:
                    print(f"❌ Failed to access search page for {term}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find document links
                # Look for various patterns of document links
                doc_links = []
                
                # Pattern 1: Direct PDF links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '.pdf' in href.lower():
                        doc_links.append({
                            'url': urljoin(search_url, href),
                            'title': link.get_text(strip=True) or 'Document',
                            'type': 'pdf'
                        })
                
                # Pattern 2: Document page links that might contain PDFs
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(keyword in href for keyword in ['/eli/', '/doc/', '/ligj']):
                        doc_links.append({
                            'url': urljoin(search_url, href),
                            'title': link.get_text(strip=True) or 'Document',
                            'type': 'page'
                        })
                
                print(f"📄 Found {len(doc_links)} potential documents for '{term}'")
                
                # Process found documents
                for i, doc in enumerate(doc_links[:self.max_docs]):
                    if len(found_documents) >= self.max_docs:
                        break
                        
                    doc_id = f"{term}_{i}"
                    
                    # Check if already processed
                    if any(d['id'] == doc_id for d in self.metadata['documents']):
                        continue
                    
                    print(f"📥 Processing: {doc['title'][:50]}...")
                    
                    if doc['type'] == 'pdf':
                        # Direct PDF download
                        filename = self.clean_filename(f"{doc_id}_{doc['title']}.pdf")
                        if self.download_pdf(doc['url'], filename):
                            found_documents.append({
                                'id': doc_id,
                                'title': doc['title'],
                                'url': doc['url'],
                                'filename': filename,
                                'type': 'pdf',
                                'search_term': term,
                                'downloaded_at': datetime.now().isoformat()
                            })
                    
                    elif doc['type'] == 'page':
                        # Try to find PDF links on the document page
                        pdf_url = self.find_pdf_on_page(doc['url'])
                        if pdf_url:
                            filename = self.clean_filename(f"{doc_id}_{doc['title']}.pdf")
                            if self.download_pdf(pdf_url, filename):
                                found_documents.append({
                                    'id': doc_id,
                                    'title': doc['title'],
                                    'url': doc['url'],
                                    'pdf_url': pdf_url,
                                    'filename': filename,
                                    'type': 'pdf',
                                    'search_term': term,
                                    'downloaded_at': datetime.now().isoformat()
                                })
                    
                    # Rate limiting
                    time.sleep(1)
                
            except Exception as e:
                print(f"❌ Search error for term '{term}': {e}")
                continue
        
        # Update metadata
        self.metadata['documents'].extend(found_documents)
        self.save_metadata()
        
        print(f"🎉 Total documents found: {len(found_documents)}")
        return found_documents
    
    def find_pdf_on_page(self, page_url: str) -> Optional[str]:
        """
        Find PDF download link on a document page
        
        Args:
            page_url: URL of the document page
            
        Returns:
            Optional[str]: PDF URL if found
        """
        try:
            response = self.session.get(page_url, timeout=20)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for PDF links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '.pdf' in href.lower() or 'download' in href.lower():
                    return urljoin(page_url, href)
            
            return None
        except:
            return None
    
    def process_downloaded_documents(self) -> List[Dict]:
        """
        Process all downloaded PDF documents and extract text
        
        Returns:
            List[Dict]: Processed document data suitable for RAG
        """
        processed_docs = []
        pdf_dir = os.path.join(self.output_dir, 'pdfs')
        
        for doc_meta in self.metadata['documents']:
            if 'processed' in doc_meta and doc_meta['processed']:
                continue  # Skip already processed
            
            pdf_path = os.path.join(pdf_dir, doc_meta['filename'])
            
            if not os.path.exists(pdf_path):
                continue
            
            print(f"📖 Processing: {doc_meta['title'][:50]}...")
            
            # Extract text from PDF
            text = self.extract_pdf_text(pdf_path)
            
            if not text or len(text) < 100:  # Skip if too short
                print(f"⚠️ Skipping {doc_meta['filename']} - insufficient text")
                continue
            
            # Process text into chunks
            processed = self.process_text(text, doc_meta['title'])
            
            # Add metadata
            processed.update({
                'id': doc_meta['id'],
                'source_url': doc_meta['url'],
                'search_term': doc_meta['search_term'],
                'filename': doc_meta['filename']
            })
            
            processed_docs.append(processed)
            
            # Save processed data
            processed_file = os.path.join(
                self.output_dir, 'processed', 
                f"{doc_meta['id']}_processed.json"
            )
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
            
            # Mark as processed
            doc_meta['processed'] = True
            
            print(f"✅ Processed: {len(processed['chunks'])} chunks, {processed['total_length']} chars")
        
        self.save_metadata()
        return processed_docs
    
    def get_processed_documents_for_rag(self) -> List[Dict]:
        """
        Get all processed documents in format suitable for RAG system
        
        Returns:
            List[Dict]: Documents formatted for RAG integration
        """
        rag_documents = []
        processed_dir = os.path.join(self.output_dir, 'processed')
        
        if not os.path.exists(processed_dir):
            return rag_documents
        
        for filename in os.listdir(processed_dir):
            if filename.endswith('_processed.json'):
                try:
                    with open(os.path.join(processed_dir, filename), 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                    
                    # Handle different document formats
                    if 'chunks' in doc_data:
                        # New format from PDF processor
                        for i, chunk_data in enumerate(doc_data['chunks']):
                            # Handle both dict and string chunks
                            if isinstance(chunk_data, dict):
                                chunk_content = chunk_data.get('content', str(chunk_data))
                            else:
                                chunk_content = str(chunk_data)
                            
                            rag_doc = {
                                'id': f"{doc_data['id']}_chunk_{i}",
                                'title': doc_data['title'],
                                'source': doc_data.get('filename', doc_data.get('source_url', 'Unknown Source')),
                                'content': chunk_content,
                                'content_en': self.translate_to_english(chunk_content),
                                'url': doc_data.get('source_url', ''),
                                'document_type': doc_data.get('document_type', 'scraped_legal_document')
                            }
                            rag_documents.append(rag_doc)
                    
                    elif isinstance(doc_data, list):
                        # Old format - list of chunks
                        for i, chunk in enumerate(doc_data):
                            rag_doc = {
                                'id': f"{filename}_chunk_{i}",
                                'title': f"Document from {filename}",
                                'source': f"qbz.gov.al - {filename}",
                                'content': chunk,
                                'content_en': self.translate_to_english(chunk),
                                'url': '',
                                'document_type': 'scraped_legal_document'
                            }
                            rag_documents.append(rag_doc)
                    
                    else:
                        # Handle legacy format with search_term (old scraped documents)
                        chunks = doc_data.get('chunks', [])
                        for i, chunk in enumerate(chunks):
                            rag_doc = {
                                'id': f"{doc_data['id']}_chunk_{i}",
                                'title': doc_data['title'],
                                'source': f"qbz.gov.al - {doc_data.get('search_term', 'Legal Document')}",
                                'content': chunk,
                                'content_en': self.translate_to_english(chunk),
                                'url': doc_data.get('source_url', ''),
                                'document_type': 'scraped_legal_document'
                            }
                            rag_documents.append(rag_doc)
                
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        print(f"📚 Loaded {len(rag_documents)} document chunks for RAG")
        return rag_documents
    
    def translate_to_english(self, text: str) -> str:
        """
        Placeholder for Albanian to English translation
        In production, integrate with translation API
        
        Args:
            text: Albanian text
            
        Returns:
            str: English translation (placeholder)
        """
        # For now, return a simple indication
        # In production, integrate with Google Translate API or similar
        return f"[English translation of: {text[:100]}...]"


def main():
    """
    Main function to demonstrate the scraper
    """
    print("🏛️ Albanian Legal Document Scraper")
    print("=" * 50)
    
    # Initialize scraper
    scraper = AlbanianLegalScraper(max_docs=5)  # Limit for testing
    
    # Scrape documents
    print("📥 Starting document scraping...")
    found_docs = scraper.scrape_qbz_search(['ligj', 'kod'])
    
    if found_docs:
        print(f"\n📖 Processing {len(found_docs)} documents...")
        processed_docs = scraper.process_downloaded_documents()
        
        print(f"\n📚 Converting to RAG format...")
        rag_docs = scraper.get_processed_documents_for_rag()
        
        print(f"\n🎉 Ready for RAG: {len(rag_docs)} document chunks")
        
        # Show sample
        if rag_docs:
            print("\n📄 Sample document chunk:")
            sample = rag_docs[0]
            print(f"Title: {sample['title']}")
            print(f"Content: {sample['content'][:200]}...")
    else:
        print("❌ No documents found")


if __name__ == "__main__":
    main()
