"""
Configuration management for AI service
Handles environment variables and settings
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Server settings
    port: int = Field(default=8000, description="Server port")
    host: str = Field(default="0.0.0.0", description="Server host")
    debug: bool = Field(default=False, description="Debug mode")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API key")
    
    # Database connections
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant vector database URL")
    
    # Model settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    max_embedding_batch_size: int = Field(default=32, description="Max batch size for embeddings")
    
    # Performance settings
    max_workers: int = Field(default=4, description="Max worker threads")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # OpenAI specific settings
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
    openai_max_tokens: int = Field(default=1000, description="Max tokens for OpenAI requests")
    openai_temperature: float = Field(default=0.3, description="OpenAI temperature")
    
    # Vector database settings
    vector_collection_size: int = Field(default=384, description="Vector embedding size")
    vector_similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    
    # Feature flags
    enable_openai: bool = Field(default=True, description="Enable OpenAI features")
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    
    if _settings is None:
        _settings = Settings()
    
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings


# Environment-specific configurations
def get_database_config() -> dict:
    """Get database configuration"""
    settings = get_settings()
    
    if not settings.database_url:
        return {}
    
    return {
        "url": settings.database_url,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600
    }


def get_redis_config() -> dict:
    """Get Redis configuration"""
    settings = get_settings()
    
    return {
        "url": settings.redis_url,
        "decode_responses": True,
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
        "retry_on_timeout": True,
        "health_check_interval": 30
    }


def get_qdrant_config() -> dict:
    """Get Qdrant configuration"""
    settings = get_settings()
    
    return {
        "url": settings.qdrant_url,
        "timeout": 30,
        "prefer_grpc": False
    }


def get_logging_config() -> dict:
    """Get logging configuration"""
    settings = get_settings()
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "default",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "": {
                "level": settings.log_level,
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    
    # Add file handler if log file is specified
    if settings.log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.log_level,
            "formatter": "detailed",
            "filename": settings.log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        
        # Add file handler to loggers
        for logger in config["loggers"].values():
            logger["handlers"].append("file")
    
    return config


def validate_settings() -> tuple[bool, list[str]]:
    """Validate current settings"""
    settings = get_settings()
    errors = []
    
    # Check required settings
    if not settings.database_url:
        errors.append("DATABASE_URL is required")
    
    # Validate OpenAI settings if enabled
    if settings.enable_openai and not settings.openai_api_key:
        errors.append("OPENAI_API_KEY is required when OpenAI features are enabled")
    
    # Validate URLs
    urls_to_check = [
        ("redis_url", settings.redis_url),
        ("qdrant_url", settings.qdrant_url)
    ]
    
    for name, url in urls_to_check:
        if not url.startswith(("http://", "https://", "redis://")):
            errors.append(f"{name} must be a valid URL")
    
    # Validate numeric ranges
    if not 1 <= settings.port <= 65535:
        errors.append("Port must be between 1 and 65535")
    
    if not 0.0 <= settings.openai_temperature <= 2.0:
        errors.append("OpenAI temperature must be between 0.0 and 2.0")
    
    if not 0.0 <= settings.vector_similarity_threshold <= 1.0:
        errors.append("Vector similarity threshold must be between 0.0 and 1.0")
    
    return len(errors) == 0, errors


def get_feature_flags() -> dict:
    """Get current feature flags"""
    settings = get_settings()
    
    return {
        "openai_enabled": settings.enable_openai and bool(settings.openai_api_key),
        "caching_enabled": settings.enable_caching,
        "monitoring_enabled": settings.enable_monitoring,
        "debug_mode": settings.debug
    }


def get_model_settings() -> dict:
    """Get ML model specific settings"""
    settings = get_settings()
    
    return {
        "embedding_model": settings.embedding_model,
        "max_batch_size": settings.max_embedding_batch_size,
        "vector_size": settings.vector_collection_size,
        "similarity_threshold": settings.vector_similarity_threshold,
        "openai_model": settings.openai_model,
        "openai_max_tokens": settings.openai_max_tokens,
        "openai_temperature": settings.openai_temperature
    }
