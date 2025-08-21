"""
Data Processing Module for Albanian Legal RAG System

This module provides comprehensive document processing capabilities including:
- Web scraping from Albanian legal sources (qbz.gov.al)
- PDF document processing and text extraction
- Document cleaning and preparation for RAG
- Unified processing pipeline for all document sources

Main Classes:
- AlbanianLegalScraper: Web scraping functionality
- PDFProcessor: PDF processing and text extraction
- DocumentProcessor: Unified processing pipeline
"""

from .processing import (
    AlbanianLegalScraper,
    PDFProcessor, 
    DocumentProcessor
)

__all__ = [
    'AlbanianLegalScraper',
    'PDFProcessor',
    'DocumentProcessor'
]
