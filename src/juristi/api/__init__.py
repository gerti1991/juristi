"""
REST API Module for Albanian Legal RAG System

This module provides a FastAPI-based REST API interface:
- Document search endpoints
- AI response generation
- System status and monitoring
- RESTful interface for all system functionality

Main Components:
- FastAPI app instance with all endpoints
- Pydantic models for request/response validation
- CORS and middleware configuration
"""

from .main import app

__all__ = ['app']
