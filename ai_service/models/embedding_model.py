"""
Embedding Model for generating vector representations
Handles text-to-vector conversion for semantic similarity
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
import numpy as np

from sentence_transformers import SentenceTransformer
import openai
from openai import AsyncOpenAI

from utils.config import get_settings

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Handles text embedding generation using multiple models
    Supports both local SentenceTransformers and OpenAI embeddings
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.local_model: Optional[SentenceTransformer] = None
        self.openai_client: Optional[AsyncOpenAI] = None
        self.model_name = "all-MiniLM-L6-v2"  # Default local model
        self.openai_model = "text-embedding-ada-002"
        self.embedding_dim = 384  # For local model
        
    async def initialize(self):
        """Initialize embedding models"""
        try:
            logger.info("Initializing embedding models...")
            
            # Initialize local SentenceTransformer model
            logger.info(f"Loading local model: {self.model_name}")
            self.local_model = SentenceTransformer(self.model_name)
            
            # Initialize OpenAI client if API key is available
            if self.settings.openai_api_key:
                logger.info("Initializing OpenAI client...")
                self.openai_client = AsyncOpenAI(
                    api_key=self.settings.openai_api_key
                )
            else:
                logger.warning("OpenAI API key not found. OpenAI embeddings will not be available.")
            
            logger.info("✅ Embedding models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding models: {e}")
            raise e
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        model_type: str = "local",
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            model_type: "local" or "openai"
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            if model_type == "openai" and self.openai_client:
                return await self._generate_openai_embeddings(texts, batch_size)
            else:
                return await self._generate_local_embeddings(texts, batch_size)
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Fallback to local model
            if model_type == "openai":
                logger.info("Falling back to local embeddings...")
                return await self._generate_local_embeddings(texts, batch_size)
            raise e
    
    async def _generate_local_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings using local SentenceTransformer model"""
        
        if not self.local_model:
            raise RuntimeError("Local embedding model not initialized")
        
        logger.info(f"Generating {len(texts)} embeddings using local model...")
        
        # Process in batches to avoid memory issues
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None, 
                self.local_model.encode, 
                batch_texts
            )
            
            # Convert to list format
            all_embeddings.extend(batch_embeddings.tolist())
        
        logger.info(f"Generated {len(all_embeddings)} local embeddings")
        return all_embeddings
    
    async def _generate_openai_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        logger.info(f"Generating {len(texts)} embeddings using OpenAI...")
        
        all_embeddings = []
        
        # Process in batches (OpenAI has rate limits)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = await self.openai_client.embeddings.create(
                    model=self.openai_model,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting
                if len(texts) > batch_size:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"OpenAI embedding batch failed: {e}")
                raise e
        
        logger.info(f"Generated {len(all_embeddings)} OpenAI embeddings")
        return all_embeddings
    
    async def generate_single_embedding(
        self, 
        text: str, 
        model_type: str = "local"
    ) -> List[float]:
        """Generate embedding for a single text"""
        
        embeddings = await self.generate_embeddings([text], model_type)
        return embeddings[0] if embeddings else []
    
    def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        return {
            "local_model": {
                "name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "loaded": self.local_model is not None
            },
            "openai_model": {
                "name": self.openai_model,
                "embedding_dim": 1536,  # OpenAI embedding dimension
                "available": self.openai_client is not None
            }
        }
    
    async def preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation"""
        
        if not text:
            return ""
        
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (for API limits)
        max_length = 8000  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"Text truncated to {max_length} characters")
        
        return text
    
    async def embed_job_data(self, job_data: Dict[str, Any]) -> List[float]:
        """Generate embedding for job data"""
        
        # Combine relevant job fields
        job_text_parts = []
        
        if job_data.get("title"):
            job_text_parts.append(f"Title: {job_data['title']}")
        
        if job_data.get("description"):
            job_text_parts.append(f"Description: {job_data['description']}")
        
        if job_data.get("skills"):
            skills_text = ", ".join(job_data["skills"])
            job_text_parts.append(f"Skills: {skills_text}")
        
        if job_data.get("requirements"):
            req_text = ". ".join(job_data["requirements"])
            job_text_parts.append(f"Requirements: {req_text}")
        
        # Combine all parts
        job_text = ". ".join(job_text_parts)
        job_text = await self.preprocess_text(job_text)
        
        return await self.generate_single_embedding(job_text)
    
    async def embed_candidate_data(self, candidate_data: Dict[str, Any]) -> List[float]:
        """Generate embedding for candidate data"""
        
        # Combine relevant candidate fields
        candidate_text_parts = []
        
        if candidate_data.get("title"):
            candidate_text_parts.append(f"Title: {candidate_data['title']}")
        
        if candidate_data.get("summary"):
            candidate_text_parts.append(f"Summary: {candidate_data['summary']}")
        
        if candidate_data.get("skills"):
            skills_text = ", ".join(candidate_data["skills"])
            candidate_text_parts.append(f"Skills: {skills_text}")
        
        if candidate_data.get("experience"):
            candidate_text_parts.append(f"Experience: {candidate_data['experience']}")
        
        if candidate_data.get("education"):
            candidate_text_parts.append(f"Education: {candidate_data['education']}")
        
        # Combine all parts
        candidate_text = ". ".join(candidate_text_parts)
        candidate_text = await self.preprocess_text(candidate_text)
        
        return await self.generate_single_embedding(candidate_text)
