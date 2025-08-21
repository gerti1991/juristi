"""
Data Processing Module for Albanian Legal RAG System

Consolidates PDF processing, web scraping, and document management functionality.
Combines logic from scraper.py and setup.py into a unified processing system.
"""

import os
import json
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

# PDF processing imports (optional)
try:
    import PyPDF2
    from docx import Document
    PDF_AVAILABLE = True
except ImportError:
    print("Warning: PDF/DOCX processing libraries not available")
    PDF_AVAILABLE = False


class AlbanianLegalScraper:
    """
    Albanian Legal Document Scraper for qbz.gov.al
    
    Handles web scraping, document downloading, and content extraction
    for Albanian legal documents.
    """
    
    def __init__(self, output_dir: str = 'legal_documents', max_docs: int = 20):
        """Initialize the scraper with configuration."""
        self.output_dir = output_dir
        self.max_docs = max_docs
        
        # Setup HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Create directories
        self._create_directories()
        
        # Load metadata
        self.metadata_file = os.path.join(output_dir, 'document_metadata.json')
        self.metadata = self._load_metadata()
    
    def _create_directories(self):
        """Create necessary output directories."""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, 'pdfs'),
            os.path.join(self.output_dir, 'processed')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """Load existing document metadata."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
        
        return {
            'documents': [],
            'last_update': None,
            'total_documents': 0
        }
    
    def _save_metadata(self):
        """Save document metadata."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def scrape_qbz_search(self, search_terms: List[str]) -> List[Dict]:
        """
        Scrape legal documents from qbz.gov.al based on search terms.
        
        Args:
            search_terms: List of terms to search for
            
        Returns:
            List of document metadata dictionaries
        """
        all_documents = []
        base_url = "https://qbz.gov.al"
        
        print(f"🔍 Searching qbz.gov.al for terms: {search_terms}")
        
        for term in search_terms:
            try:
                # Construct search URL
                search_url = f"{base_url}/eli/search"
                params = {
                    'q': term,
                    'type': 'all',
                    'limit': min(50, self.max_docs)
                }
                
                print(f"   📋 Searching for: '{term}'")
                response = self.session.get(search_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    documents = self._parse_search_results(response.text, base_url)
                    all_documents.extend(documents)
                    print(f"   ✅ Found {len(documents)} documents for '{term}'")
                else:
                    print(f"   ❌ Search failed for '{term}': {response.status_code}")
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Error searching for '{term}': {e}")
        
        # Remove duplicates based on URL
        unique_docs = {}
        for doc in all_documents:
            url = doc.get('url', '')
            if url and url not in unique_docs:
                unique_docs[url] = doc
        
        unique_documents = list(unique_docs.values())
        print(f"📊 Total unique documents found: {len(unique_documents)}")
        
        return unique_documents[:self.max_docs]
    
    def _parse_search_results(self, html: str, base_url: str) -> List[Dict]:
        """Parse search results HTML to extract document information."""
        documents = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for document links (adjust selectors based on actual site structure)
            result_items = soup.find_all(['article', 'div'], class_=re.compile(r'result|item|document'))
            
            for item in result_items:
                try:
                    # Extract title
                    title_elem = item.find(['h1', 'h2', 'h3', 'h4', 'a'])
                    title = title_elem.get_text().strip() if title_elem else "Unknown Document"
                    
                    # Extract URL
                    link_elem = item.find('a', href=True)
                    if link_elem:
                        url = urljoin(base_url, link_elem['href'])
                    else:
                        continue
                    
                    # Extract description/summary if available
                    desc_elem = item.find(['p', 'div'], class_=re.compile(r'desc|summary|content'))
                    description = desc_elem.get_text().strip() if desc_elem else ""
                    
                    # Extract date if available
                    date_elem = item.find(text=re.compile(r'\d{2}[./]\d{2}[./]\d{4}'))
                    date = date_elem.strip() if date_elem else ""
                    
                    document = {
                        'title': title,
                        'url': url,
                        'description': description,
                        'date': date,
                        'scraped_at': datetime.now().isoformat(),
                        'source': 'qbz.gov.al'
                    }
                    
                    documents.append(document)
                    
                except Exception as e:
                    print(f"Warning: Could not parse document item: {e}")
                    continue
            
        except Exception as e:
            print(f"Warning: Could not parse search results: {e}")
        
        return documents
    
    def download_document_content(self, doc: Dict) -> Optional[str]:
        """
        Download and extract content from a legal document.
        
        Args:
            doc: Document metadata dictionary
            
        Returns:
            Extracted text content or None
        """
        url = doc.get('url')
        if not url:
            return None
        
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # If it's an HTML page, extract text
                if 'text/html' in response.headers.get('content-type', ''):
                    return self._extract_html_content(response.text)
                
                # If it's a PDF, try to extract text
                elif 'pdf' in response.headers.get('content-type', ''):
                    return self._extract_pdf_content(response.content)
                
                # If it's plain text
                elif 'text/plain' in response.headers.get('content-type', ''):
                    return response.text
                
                else:
                    # Try to extract as HTML anyway
                    return self._extract_html_content(response.text)
            
            else:
                print(f"Warning: Could not download {url}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Warning: Error downloading {url}: {e}")
            return None
    
    def _extract_html_content(self, html: str) -> str:
        """Extract clean text content from HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Look for main content areas
            content_selectors = [
                'main', '.content', '#content', 'article', 
                '.document-content', '.law-content', '.legal-text'
            ]
            
            content = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text()
                    break
            
            # Fallback to body content
            if not content:
                content = soup.get_text()
            
            # Clean up the text
            content = re.sub(r'\s+', ' ', content).strip()
            content = re.sub(r'\n\s*\n', '\n\n', content)
            
            return content
            
        except Exception as e:
            print(f"Warning: Could not extract HTML content: {e}")
            return ""
    
    def _extract_pdf_content(self, pdf_data: bytes) -> str:
        """Extract text content from PDF data."""
        if not PDF_AVAILABLE:
            return ""
        
        try:
            import io
            pdf_file = io.BytesIO(pdf_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            print(f"Warning: Could not extract PDF content: {e}")
            return ""
    
    def process_downloaded_documents(self) -> List[Dict]:
        """Process downloaded documents and prepare for RAG."""
        processed_docs = []
        
        for doc in self.metadata.get('documents', []):
            try:
                # Download content if not already done
                if 'content' not in doc or not doc['content']:
                    content = self.download_document_content(doc)
                    if content:
                        doc['content'] = content
                
                # Process for RAG if content exists
                if doc.get('content'):
                    rag_doc = self._prepare_for_rag(doc)
                    if rag_doc:
                        processed_docs.append(rag_doc)
                        
            except Exception as e:
                print(f"Warning: Error processing document {doc.get('title', 'Unknown')}: {e}")
        
        return processed_docs
    
    def _prepare_for_rag(self, doc: Dict) -> Optional[Dict]:
        """Prepare document for RAG system."""
        content = doc.get('content', '').strip()
        if len(content) < 100:  # Skip very short documents
            return None
        
        # Clean content
        content = self._clean_legal_text(content)
        
        # Create RAG document
        rag_doc = {
            'id': f"scraped_{hash(doc.get('url', ''))}",
            'title': doc.get('title', 'Unknown Document'),
            'content': content,
            'source': doc.get('url', 'Unknown'),
            'document_type': 'scraped_legal_document',
            'scraped_date': doc.get('scraped_at'),
            'original_date': doc.get('date', ''),
            'description': doc.get('description', '')
        }
        
        return rag_doc
    
    def _clean_legal_text(self, text: str) -> str:
        """Clean and normalize legal text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR/extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'(\d+)([A-Z])', r'\1 \2', text)    # Space after numbers
        
        # Normalize Albanian characters
        replacements = {
            'ë': 'ë', 'ç': 'ç', 'ï': 'ï', 'ü': 'ü',  # Normalize encoding
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove headers/footers patterns
        text = re.sub(r'Faqe \d+ nga \d+', '', text)
        text = re.sub(r'www\.\w+\.\w+', '', text)
        
        return text.strip()
    
    def get_processed_documents_for_rag(self) -> List[Dict]:
        """Get processed documents ready for RAG system."""
        rag_file = os.path.join(self.output_dir, 'processed_rag_documents.json')
        
        if os.path.exists(rag_file):
            try:
                with open(rag_file, 'r', encoding='utf-8') as f:
                    rag_documents = json.load(f)
                print(f"📚 Loaded {len(rag_documents)} document chunks for RAG")
                return rag_documents
            except Exception as e:
                print(f"Warning: Could not load RAG documents: {e}")
        
        return []
    
    def save_processed_documents(self, documents: List[Dict]):
        """Save processed documents for RAG system."""
        rag_file = os.path.join(self.output_dir, 'processed_rag_documents.json')
        
        try:
            with open(rag_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Saved {len(documents)} processed documents")
            
        except Exception as e:
            print(f"Error: Could not save processed documents: {e}")


class PDFProcessor:
    """
    PDF Processing System for Albanian Legal Documents
    
    Handles PDF parsing, text extraction, and document structuring
    for the RAG system.
    """
    
    def __init__(self, pdf_directory: str = "legal_documents/pdfs"):
        """Initialize PDF processor."""
        self.pdf_directory = pdf_directory
        self.output_directory = "legal_documents/processed"
        
        # Create directories
        os.makedirs(self.pdf_directory, exist_ok=True)
        os.makedirs(self.output_directory, exist_ok=True)
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDFs in the directory."""
        if not PDF_AVAILABLE:
            print("Error: PDF processing libraries not available")
            return []
        
        pdf_files = []
        if os.path.exists(self.pdf_directory):
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found for processing")
            return []
        
        print(f"📄 Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        
        for pdf_file in pdf_files:
            print(f"   Processing: {pdf_file}")
            try:
                documents = self.process_single_pdf(pdf_file)
                all_documents.extend(documents)
                print(f"   ✅ Extracted {len(documents)} chunks")
            except Exception as e:
                print(f"   ❌ Error processing {pdf_file}: {e}")
        
        # Save processed documents
        self.save_processed_documents(all_documents)
        
        return all_documents
    
    def process_single_pdf(self, filename: str) -> List[Dict]:
        """Process a single PDF file."""
        if not PDF_AVAILABLE:
            return []
        
        filepath = os.path.join(self.pdf_directory, filename)
        documents = []
        
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = pdf_reader.metadata if pdf_reader.metadata else {}
                title = metadata.get('/Title', filename.replace('.pdf', ''))
                
                # Extract text from all pages
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        full_text += f"\n[Faqe {page_num + 1}]\n{page_text}"
                
                if full_text.strip():
                    # Split into manageable chunks
                    chunks = self._split_into_chunks(full_text, title, filename)
                    documents.extend(chunks)
        
        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")
        
        return documents
    
    def _split_into_chunks(self, text: str, title: str, filename: str, chunk_size: int = 1000) -> List[Dict]:
        """Split text into chunks for RAG processing."""
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_size = 0
        chunk_index = 1
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= chunk_size:
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunk = {
                        'id': f'pdf_{filename}_{chunk_index}',
                        'title': f'{title} - Part {chunk_index}',
                        'content': chunk_text,
                        'source': filename,
                        'document_type': 'pdf_chunk',
                        'chunk_index': chunk_index,
                        'parent_document': title
                    }
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = []
                current_size = 0
        
        # Handle remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunk = {
                    'id': f'pdf_{filename}_{chunk_index}',
                    'title': f'{title} - Part {chunk_index}',
                    'content': chunk_text,
                    'source': filename,
                    'document_type': 'pdf_chunk',
                    'chunk_index': chunk_index,
                    'parent_document': title
                }
                chunks.append(chunk)
        
        return chunks
    
    def save_processed_documents(self, documents: List[Dict]):
        """Save processed PDF documents."""
        output_file = os.path.join(self.output_directory, 'pdf_documents.json')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Saved {len(documents)} PDF chunks to {output_file}")
            
        except Exception as e:
            print(f"Error: Could not save PDF documents: {e}")


class DocumentProcessor:
    """
    Unified Document Processing System
    
    Combines scraping and PDF processing for comprehensive document management.
    """
    
    def __init__(self, output_dir: str = "legal_documents"):
        """Initialize the unified processor."""
        self.output_dir = output_dir
        self.scraper = AlbanianLegalScraper(output_dir)
        self.pdf_processor = PDFProcessor(os.path.join(output_dir, "pdfs"))
    
    def process_all_sources(self, search_terms: List[str] = None) -> Dict[str, List]:
        """Process documents from all sources."""
        results = {
            'scraped': [],
            'pdfs': [],
            'total': 0
        }
        
        print("🚀 Starting comprehensive document processing...")
        
        # Process web scraped documents
        if search_terms:
            print("\n📡 Processing web scraped documents...")
            try:
                found_docs = self.scraper.scrape_qbz_search(search_terms)
                processed_docs = self.scraper.process_downloaded_documents()
                results['scraped'] = processed_docs
                print(f"✅ Web scraping completed: {len(processed_docs)} documents")
            except Exception as e:
                print(f"❌ Web scraping failed: {e}")
        
        # Process PDF documents
        print("\n📄 Processing PDF documents...")
        try:
            pdf_docs = self.pdf_processor.process_all_pdfs()
            results['pdfs'] = pdf_docs
            print(f"✅ PDF processing completed: {len(pdf_docs)} documents")
        except Exception as e:
            print(f"❌ PDF processing failed: {e}")
        
        # Calculate totals
        results['total'] = len(results['scraped']) + len(results['pdfs'])
        
        print(f"\n🎯 Document processing summary:")
        print(f"   📡 Web scraped: {len(results['scraped'])}")
        print(f"   📄 PDF processed: {len(results['pdfs'])}")
        print(f"   📊 Total documents: {results['total']}")
        
        return results


# Export main classes
__all__ = ['AlbanianLegalScraper', 'PDFProcessor', 'DocumentProcessor']
