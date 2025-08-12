"""
AI Job Matching Service - Main Entry Point
FastAPI application for advanced AI/ML features
"""

import os
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import embeddings, analysis, matching, health
from utils.config import get_settings
from utils.database import init_database
from services.vector_service import VectorService
from models.embedding_model import EmbeddingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for services
vector_service = None
embedding_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global vector_service, embedding_model
    
    logger.info("🚀 Starting AI Job Matching Service...")
    
    try:
        # Initialize configuration
        settings = get_settings()
        
        # Initialize database connections
        await init_database()
        
        # Initialize ML models
        logger.info("Loading ML models...")
        embedding_model = EmbeddingModel()
        await embedding_model.initialize()
        
        # Initialize vector service
        logger.info("Initializing vector service...")
        vector_service = VectorService(settings.qdrant_url)
        await vector_service.initialize()
        
        # Store in app state
        app.state.vector_service = vector_service
        app.state.embedding_model = embedding_model
        app.state.settings = settings
        
        logger.info("✅ AI Service startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start AI service: {e}")
        raise e
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down AI Service...")
    if vector_service:
        await vector_service.close()
    logger.info("✅ AI Service shutdown completed")

# Create FastAPI app
app = FastAPI(
    title="AI Job Matching Service",
    description="Advanced AI/ML features for job matching system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(embeddings.router, prefix="/embeddings", tags=["Embeddings"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(matching.router, prefix="/matching", tags=["Matching"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AI Job Matching Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    # Get settings
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )
