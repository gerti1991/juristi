"""
User Interface Module for Albanian Legal RAG System

This module provides web-based user interfaces for the system:
- Streamlit-based web interface
- Interactive search and results display
- Configuration and settings management

Main Components:
- StreamlitInterface: Main web application interface
- run_streamlit_app: Entry point for Streamlit app
"""

from .main import StreamlitInterface, run_streamlit_app

__all__ = ['StreamlitInterface', 'run_streamlit_app']
