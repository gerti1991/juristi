"""
Configuration Management for Albanian Legal RAG System

Centralizes all configuration settings with validation and defaults.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    chroma_db_path: str = "./chroma_db"
    documents_path: str = "./legal_documents/pdfs"
    index_file: str = "document_index.json"


@dataclass
class ModelConfig:
    """Model and embedding configuration settings."""
    embedding_provider: str = "bge"  # google, bge, sentence-transformers
    gemini_model: str = "gemini-2.5-flash"
    bge_model: str = "BAAI/bge-small-en-v1.5"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class ProcessingConfig:
    """Document processing configuration settings."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_docs_per_query: int = 15
    similarity_threshold: float = 0.3
    mmr_diversity_score: float = 0.3


@dataclass
class UIConfig:
    """User interface configuration settings."""
    default_query_mode: str = "precise"  # precise, analyzed
    show_sources: bool = True
    verbose_mode: bool = True
    page_title: str = "Juristi AI - Albanian Legal Assistant"
    page_icon: str = "⚖️"


@dataclass
class APIConfig:
    """API server configuration settings."""
    host: str = "localhost"
    port: int = 8000
    reload: bool = True
    cors_origins: list = None


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: Optional[str] = None


class ConfigManager:
    """Central configuration manager with environment variable support."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self._load_env_file()
        self.database = self._load_database_config()
        self.model = self._load_model_config()
        self.processing = self._load_processing_config()
        self.ui = self._load_ui_config()
        self.api = self._load_api_config()
        self.logging = self._load_logging_config()
    
    def _load_env_file(self):
        """Load environment variables from .env file if it exists."""
        env_file = Path(".env")
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                # python-dotenv not installed, skip
                pass
    
    def _get_env(self, key: str, default: Any = None, type_func=str) -> Any:
        """Get environment variable with type conversion and default."""
        value = os.getenv(key)
        if value is None:
            return default
        
        try:
            if type_func == bool:
                return value.lower() in ('true', '1', 'yes', 'on')
            elif type_func == list:
                return [item.strip() for item in value.split(',') if item.strip()]
            else:
                return type_func(value)
        except (ValueError, TypeError):
            return default
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration."""
        return DatabaseConfig(
            chroma_db_path=self._get_env("CHROMA_DB_PATH", "./chroma_db"),
            documents_path=self._get_env("DOCUMENTS_PATH", "./legal_documents/pdfs"),
            index_file=self._get_env("INDEX_FILE", "document_index.json")
        )
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration."""
        return ModelConfig(
            embedding_provider=self._get_env("EMBEDDING_PROVIDER", "bge"),
            gemini_model=self._get_env("GEMINI_MODEL", "gemini-2.5-flash"),
            bge_model=self._get_env("BGE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            sentence_transformer_model=self._get_env("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
            device=self._get_env("DEVICE", "auto")
        )
    
    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration."""
        return ProcessingConfig(
            chunk_size=self._get_env("CHUNK_SIZE", 1000, int),
            chunk_overlap=self._get_env("CHUNK_OVERLAP", 200, int),
            max_docs_per_query=self._get_env("MAX_DOCS_PER_QUERY", 15, int),
            similarity_threshold=self._get_env("SIMILARITY_THRESHOLD", 0.3, float),
            mmr_diversity_score=self._get_env("MMR_DIVERSITY_SCORE", 0.3, float)
        )
    
    def _load_ui_config(self) -> UIConfig:
        """Load UI configuration."""
        return UIConfig(
            default_query_mode=self._get_env("DEFAULT_QUERY_MODE", "precise"),
            show_sources=self._get_env("SHOW_SOURCES", True, bool),
            verbose_mode=self._get_env("VERBOSE_MODE", True, bool),
            page_title=self._get_env("PAGE_TITLE", "Juristi AI - Albanian Legal Assistant"),
            page_icon=self._get_env("PAGE_ICON", "⚖️")
        )
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration."""
        return APIConfig(
            host=self._get_env("API_HOST", "localhost"),
            port=self._get_env("API_PORT", 8000, int),
            reload=self._get_env("API_RELOAD", True, bool),
            cors_origins=self._get_env("CORS_ORIGINS", None, list)
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration."""
        return LoggingConfig(
            level=self._get_env("LOG_LEVEL", "INFO"),
            format=self._get_env("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_handler=self._get_env("LOG_FILE", None)
        )
    
    def setup_logging(self):
        """Set up logging based on configuration."""
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format
        )
        
        if self.logging.file_handler:
            file_handler = logging.FileHandler(self.logging.file_handler)
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            logging.getLogger().addHandler(file_handler)
    
    def get_google_api_key(self) -> Optional[str]:
        """Get Google API key from environment."""
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    def has_google_credentials(self) -> bool:
        """Check if Google credentials are available."""
        return bool(self.get_google_api_key())
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        status = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check database paths
        if not Path(self.database.chroma_db_path).parent.exists():
            status["warnings"].append(f"ChromaDB parent directory does not exist: {self.database.chroma_db_path}")
        
        if not Path(self.database.documents_path).exists():
            status["warnings"].append(f"Documents directory does not exist: {self.database.documents_path}")
        
        # Check model configuration
        valid_providers = ["google", "bge", "sentence-transformers"]
        if self.model.embedding_provider not in valid_providers:
            status["errors"].append(f"Invalid embedding provider: {self.model.embedding_provider}")
            status["valid"] = False
        
        # Check device configuration
        valid_devices = ["auto", "cpu", "cuda"]
        if self.model.device not in valid_devices:
            status["warnings"].append(f"Unknown device setting: {self.model.device}")
        
        # Check query mode
        valid_modes = ["precise", "analyzed"]
        if self.ui.default_query_mode not in valid_modes:
            status["warnings"].append(f"Unknown query mode: {self.ui.default_query_mode}")
        
        return status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "database": self.database.__dict__,
            "model": self.model.__dict__,
            "processing": self.processing.__dict__,
            "ui": self.ui.__dict__,
            "api": self.api.__dict__,
            "logging": self.logging.__dict__
        }


# Global configuration instance
config = ConfigManager()
