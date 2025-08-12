"""
Vector Service for managing vector database operations
Handles storage and retrieval of embeddings in Qdrant
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    MatchValue, SearchRequest, UpdateStatus, CollectionInfo
)
from qdrant_client.http.exceptions import UnexpectedResponse

from utils.config import get_settings

logger = logging.getLogger(__name__)

class VectorService:
    """
    Service for managing vector database operations
    Provides high-level interface for Qdrant vector database
    """
    
    def __init__(self, qdrant_url: str = None):
        self.settings = get_settings()
        self.qdrant_url = qdrant_url or self.settings.qdrant_url
        self.client: Optional[QdrantClient] = None
        
        # Collection configurations
        self.collections_config = {
            "jobs": {
                "vector_size": 384,  # For sentence-transformers
                "distance": Distance.COSINE
            },
            "candidates": {
                "vector_size": 384,
                "distance": Distance.COSINE
            },
            "job_descriptions": {
                "vector_size": 1536,  # For OpenAI embeddings
                "distance": Distance.COSINE
            }
        }
    
    async def initialize(self):
        """Initialize vector database connection and collections"""
        try:
            logger.info(f"Connecting to Qdrant at {self.qdrant_url}")
            
            # Initialize client
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Test connection
            collections = await self._get_collections()
            logger.info(f"Connected to Qdrant. Found {len(collections)} existing collections")
            
            # Initialize collections
            await self._initialize_collections()
            
            logger.info("✅ Vector service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector service: {e}")
            raise e
    
    async def _get_collections(self) -> List[str]:
        """Get list of existing collections"""
        try:
            collections_response = self.client.get_collections()
            return [collection.name for collection in collections_response.collections]
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return []
    
    async def _initialize_collections(self):
        """Initialize required collections if they don't exist"""
        
        existing_collections = await self._get_collections()
        
        for collection_name, config in self.collections_config.items():
            if collection_name not in existing_collections:
                await self._create_collection(collection_name, config)
            else:
                logger.info(f"Collection '{collection_name}' already exists")
    
    async def _create_collection(self, collection_name: str, config: Dict[str, Any]):
        """Create a new collection with specified configuration"""
        
        try:
            logger.info(f"Creating collection '{collection_name}'...")
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=config["vector_size"],
                    distance=config["distance"]
                )
            )
            
            logger.info(f"✅ Collection '{collection_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise e
    
    async def store_job_embedding(
        self, 
        job_id: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> bool:
        """Store job embedding in vector database"""
        
        try:
            point = PointStruct(
                id=job_id,
                vector=embedding,
                payload={
                    **metadata,
                    "type": "job",
                    "created_at": datetime.now().isoformat(),
                    "vector_model": "sentence-transformers"
                }
            )
            
            result = self.client.upsert(
                collection_name="jobs",
                points=[point]
            )
            
            if result.status == UpdateStatus.COMPLETED:
                logger.info(f"Stored embedding for job {job_id}")
                return True
            else:
                logger.error(f"Failed to store job embedding: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing job embedding: {e}")
            return False
    
    async def store_candidate_embedding(
        self, 
        candidate_id: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> bool:
        """Store candidate embedding in vector database"""
        
        try:
            point = PointStruct(
                id=candidate_id,
                vector=embedding,
                payload={
                    **metadata,
                    "type": "candidate",
                    "created_at": datetime.now().isoformat(),
                    "vector_model": "sentence-transformers"
                }
            )
            
            result = self.client.upsert(
                collection_name="candidates",
                points=[point]
            )
            
            if result.status == UpdateStatus.COMPLETED:
                logger.info(f"Stored embedding for candidate {candidate_id}")
                return True
            else:
                logger.error(f"Failed to store candidate embedding: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing candidate embedding: {e}")
            return False
    
    async def search_similar_candidates(
        self, 
        job_embedding: List[float], 
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for candidates similar to job requirements"""
        
        try:
            # Build filter conditions
            search_filter = None
            if filters:
                conditions = []
                
                if "skills" in filters:
                    # Filter by skills (if stored in payload)
                    for skill in filters["skills"]:
                        conditions.append(
                            FieldCondition(
                                key="skills",
                                match=MatchValue(value=skill)
                            )
                        )
                
                if "location" in filters:
                    conditions.append(
                        FieldCondition(
                            key="location",
                            match=MatchValue(value=filters["location"])
                        )
                    )
                
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform vector search
            search_results = self.client.search(
                collection_name="candidates",
                query_vector=job_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter
            )
            
            # Format results
            candidates = []
            for result in search_results:
                candidates.append({
                    "candidate_id": result.id,
                    "similarity_score": result.score,
                    "metadata": result.payload
                })
            
            logger.info(f"Found {len(candidates)} similar candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Error searching candidates: {e}")
            return []
    
    async def search_similar_jobs(
        self, 
        candidate_embedding: List[float], 
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for jobs similar to candidate profile"""
        
        try:
            # Build filter conditions
            search_filter = None
            if filters:
                conditions = []
                
                if "job_type" in filters:
                    conditions.append(
                        FieldCondition(
                            key="job_type",
                            match=MatchValue(value=filters["job_type"])
                        )
                    )
                
                if "salary_min" in filters:
                    # Note: This would require range filtering in production
                    pass
                
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform vector search
            search_results = self.client.search(
                collection_name="jobs",
                query_vector=candidate_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter
            )
            
            # Format results
            jobs = []
            for result in search_results:
                jobs.append({
                    "job_id": result.id,
                    "similarity_score": result.score,
                    "metadata": result.payload
                })
            
            logger.info(f"Found {len(jobs)} similar jobs")
            return jobs
            
        except Exception as e:
            logger.error(f"Error searching jobs: {e}")
            return []
    
    async def get_job_embedding(self, job_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Retrieve job embedding and metadata"""
        
        try:
            result = self.client.retrieve(
                collection_name="jobs",
                ids=[job_id]
            )
            
            if result:
                point = result[0]
                return point.vector, point.payload
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving job embedding: {e}")
            return None
    
    async def get_candidate_embedding(self, candidate_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Retrieve candidate embedding and metadata"""
        
        try:
            result = self.client.retrieve(
                collection_name="candidates",
                ids=[candidate_id]
            )
            
            if result:
                point = result[0]
                return point.vector, point.payload
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving candidate embedding: {e}")
            return None
    
    async def delete_job_embedding(self, job_id: str) -> bool:
        """Delete job embedding from vector database"""
        
        try:
            result = self.client.delete(
                collection_name="jobs",
                points_selector=[job_id]
            )
            
            return result.status == UpdateStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Error deleting job embedding: {e}")
            return False
    
    async def delete_candidate_embedding(self, candidate_id: str) -> bool:
        """Delete candidate embedding from vector database"""
        
        try:
            result = self.client.delete(
                collection_name="candidates",
                points_selector=[candidate_id]
            )
            
            return result.status == UpdateStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Error deleting candidate embedding: {e}")
            return False
    
    async def batch_store_embeddings(
        self, 
        collection_name: str, 
        embeddings_data: List[Dict[str, Any]]
    ) -> int:
        """Store multiple embeddings in batch for better performance"""
        
        try:
            points = []
            for data in embeddings_data:
                point = PointStruct(
                    id=data["id"],
                    vector=data["embedding"],
                    payload={
                        **data.get("metadata", {}),
                        "created_at": datetime.now().isoformat()
                    }
                )
                points.append(point)
            
            # Process in chunks to avoid memory issues
            chunk_size = 100
            stored_count = 0
            
            for i in range(0, len(points), chunk_size):
                chunk = points[i:i + chunk_size]
                
                result = self.client.upsert(
                    collection_name=collection_name,
                    points=chunk
                )
                
                if result.status == UpdateStatus.COMPLETED:
                    stored_count += len(chunk)
                else:
                    logger.error(f"Failed to store batch chunk: {result}")
            
            logger.info(f"Stored {stored_count}/{len(points)} embeddings in batch")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error in batch storage: {e}")
            return 0
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection"""
        
        try:
            info = self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of vector service"""
        
        try:
            if not self.client:
                return {"status": "error", "message": "Client not initialized"}
            
            # Test connection by getting collections
            collections = await self._get_collections()
            
            # Get stats for each collection
            collection_stats = {}
            for collection_name in self.collections_config.keys():
                if collection_name in collections:
                    info = await self.get_collection_info(collection_name)
                    collection_stats[collection_name] = info
            
            return {
                "status": "healthy",
                "qdrant_url": self.qdrant_url,
                "collections": collections,
                "collection_stats": collection_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def create_backup(self, collection_name: str) -> Dict[str, Any]:
        """Create backup of collection (simplified implementation)"""
        
        try:
            # This is a simplified backup - in production you'd want
            # to use Qdrant's snapshot functionality
            
            info = await self.get_collection_info(collection_name)
            if not info:
                return {"success": False, "message": "Collection not found"}
            
            backup_info = {
                "collection_name": collection_name,
                "backup_time": datetime.now().isoformat(),
                "points_count": info["points_count"],
                "vectors_count": info["vectors_count"]
            }
            
            logger.info(f"Backup info created for {collection_name}")
            return {"success": True, "backup_info": backup_info}
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {"success": False, "message": str(e)}
    
    async def close(self):
        """Close vector service connections"""
        try:
            if self.client:
                # Qdrant client doesn't need explicit closing in current version
                # but we'll log the shutdown
                logger.info("Vector service connections closed")
        except Exception as e:
            logger.error(f"Error closing vector service: {e}")
    
    async def optimize_collections(self):
        """Optimize collections for better performance"""
        try:
            for collection_name in self.collections_config.keys():
                try:
                    # This would trigger collection optimization in Qdrant
                    # Currently just a placeholder for the API call
                    logger.info(f"Optimizing collection {collection_name}")
                    # self.client.optimize_collection(collection_name)
                except Exception as e:
                    logger.error(f"Failed to optimize {collection_name}: {e}")
        except Exception as e:
            logger.error(f"Collection optimization failed: {e}")
