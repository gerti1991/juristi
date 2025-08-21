"""
REST API Module for Albanian Legal RAG System

This module provides a FastAPI-based REST API interface:
- Document search endpoints
- AI response generation
- System status and monitoring
- RESTful interface for all system functionality

Main Components:
- AlbanianLegalAPI: Main API class with all endpoints
- Pydantic models for request/response validation
- CORS and middleware configuration
"""

from .main import AlbanianLegalAPI, app, create_app

__all__ = ['AlbanianLegalAPI', 'app', 'create_app']
