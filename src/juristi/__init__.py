"""
Albanian Legal RAG System

A comprehensive legal research assistant for Albanian law using advanced RAG techniques.
"""

__version__ = "1.0.0"
__author__ = "Albanian Legal RAG Team"

from .core.rag_engine import AlbanianLegalRAG
from .core.llm_client import CloudLLMClient

__all__ = ["AlbanianLegalRAG", "CloudLLMClient"]
