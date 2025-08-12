"""
Matching Service - High-level orchestration of matching operations
Combines multiple AI models and services for comprehensive job-candidate matching
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from models.matching_model import MatchingModel
from models.embedding_model import EmbeddingModel
from services.vector_service import VectorService
from services.nlp_service import NLPService
from services.openai_service import OpenAIService
from utils.config import get_settings

logger = logging.getLogger(__name__)

class MatchingService:
    """
    High-level service that orchestrates the complete matching process
    Combines embeddings, NLP analysis, and advanced matching algorithms
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize component services
        self.matching_model: Optional[MatchingModel] = None
        self.embedding_model: Optional[EmbeddingModel] = None
        self.vector_service: Optional[VectorService] = None
        self.nlp_service: Optional[NLPService] = None
        self.openai_service: Optional[OpenAIService] = None
        
        # Caching for performance
        self._embedding_cache = {}
        self._match_cache = {}
        
"""
Matching Service - High-level orchestration of matching operations
Combines multiple AI models and services for comprehensive job-candidate matching
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from models.matching_model import MatchingModel
from models.embedding_model import EmbeddingModel
from services.vector_service import VectorService
from services.nlp_service import NLPService
from services.openai_service import OpenAIService
from utils.config import get_settings

logger = logging.getLogger(__name__)

class MatchingService:
    """
    High-level service that orchestrates the complete matching process
    Combines embeddings, NLP analysis, and advanced matching algorithms
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize component services
        self.matching_model: Optional[MatchingModel] = None
        self.embedding_model: Optional[EmbeddingModel] = None
        self.vector_service: Optional[VectorService] = None
        self.nlp_service: Optional[NLPService] = None
        self.openai_service: Optional[OpenAIService] = None
        
        # Caching for performance
        self._embedding_cache = {}
        self._match_cache = {}
        
    async def initialize(self):
        """Initialize all component services"""
        try:
            logger.info("Initializing Matching Service...")
            
            # Initialize models
            self.matching_model = MatchingModel()
            self.embedding_model = EmbeddingModel()
            await self.embedding_model.initialize()
            
            # Initialize services
            self.vector_service = VectorService()
            await self.vector_service.initialize()
            
            self.nlp_service = NLPService()
            await self.nlp_service.initialize()
            
            self.openai_service = OpenAIService()
            await self.openai_service.initialize()
            
            logger.info("✅ Matching Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Matching Service: {e}")
            raise e
    
    async def find_candidates_for_job(
        self, 
        job_data: Dict[str, Any],
        limit: int = 20,
        min_score: float = 60.0,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Find and rank candidates for a specific job
        Uses multiple matching strategies and combines results
        """
        try:
            job_id = job_data.get("id", "unknown")
            logger.info(f"Finding candidates for job {job_id}")
            
            # Step 1: Generate job embedding
            job_embedding = await self._get_or_generate_job_embedding(job_data)
            
            # Step 2: Vector search for similar candidates
            vector_candidates = await self._vector_search_candidates(
                job_embedding, limit * 2  # Get more for filtering
            )
            
            # Step 3: Calculate detailed match scores
            detailed_matches = await self._calculate_detailed_matches(
                job_data, vector_candidates, job_embedding
            )
            
            # Step 4: Filter and rank results
            filtered_matches = [
                match for match in detailed_matches 
                if match["overall_score"] >= min_score
            ]
            
            # Sort by score
            filtered_matches.sort(key=lambda x: x["overall_score"], reverse=True)
            
            # Limit results
            final_matches = filtered_matches[:limit]
            
            # Step 5: Add recommendations if requested
            if include_recommendations and self.openai_service:
                for match in final_matches[:5]:  # Only for top 5 to save API calls
                    try:
                        recommendations = await self.openai_service.generate_candidate_recommendations(
                            job_data, match.get("candidate_data", {}), match["overall_score"]
                        )
                        match["ai_recommendations"] = recommendations
                    except Exception as e:
                        logger.error(f"Failed to generate recommendations: {e}")
                        match["ai_recommendations"] = {"error": str(e)}
            
            return {
                "job_id": job_id,
                "total_candidates_found": len(filtered_matches),
                "candidates_returned": len(final_matches),
                "min_score_threshold": min_score,
                "matches": final_matches,
                "search_metadata": {
                    "vector_candidates_found": len(vector_candidates),
                    "processed_at": datetime.now().isoformat(),
                    "processing_time_ms": 0  # Would be calculated in real implementation
                }
            }
            
        except Exception as e:
            logger.error(f"Error finding candidates for job: {e}")
            return {
                "job_id": job_data.get("id", "unknown"),
                "error": str(e),
                "matches": []
            }
    
    async def find_jobs_for_candidate(
        self, 
        candidate_data: Dict[str, Any],
        limit: int = 20,
        min_score: float = 60.0,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Find and rank jobs for a specific candidate
        Uses multiple matching strategies and combines results
        """
        try:
            candidate_id = candidate_data.get("id", "unknown")
            logger.info(f"Finding jobs for candidate {candidate_id}")
            
            # Step 1: Generate candidate embedding
            candidate_embedding = await self._get_or_generate_candidate_embedding(candidate_data)
            
            # Step 2: Vector search for similar jobs
            vector_jobs = await self._vector_search_jobs(
                candidate_embedding, limit * 2  # Get more for filtering
            )
            
            # Step 3: Calculate detailed match scores
            detailed_matches = await self._calculate_detailed_job_matches(
                candidate_data, vector_jobs, candidate_embedding
            )
            
            # Step 4: Filter and rank results
            filtered_matches = [
                match for match in detailed_matches 
                if match["overall_score"] >= min_score
            ]
            
            # Sort by score
            filtered_matches.sort(key=lambda x: x["overall_score"], reverse=True)
            
            # Limit results
            final_matches = filtered_matches[:limit]
            
            # Step 5: Add recommendations if requested
            if include_recommendations and self.openai_service:
                for match in final_matches[:5]:  # Only for top 5
                    try:
                        recommendations = await self.openai_service.generate_candidate_recommendations(
                            match.get("job_data", {}), candidate_data, match["overall_score"]
                        )
                        match["ai_recommendations"] = recommendations
                    except Exception as e:
                        logger.error(f"Failed to generate recommendations: {e}")
                        match["ai_recommendations"] = {"error": str(e)}
            
            return {
                "candidate_id": candidate_id,
                "total_jobs_found": len(filtered_matches),
                "jobs_returned": len(final_matches),
                "min_score_threshold": min_score,
                "matches": final_matches,
                "search_metadata": {
                    "vector_jobs_found": len(vector_jobs),
                    "processed_at": datetime.now().isoformat(),
                    "processing_time_ms": 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error finding jobs for candidate: {e}")
            return {
                "candidate_id": candidate_data.get("id", "unknown"),
                "error": str(e),
                "matches": []
            }
    
    async def calculate_single_match(
        self, 
        job_data: Dict[str, Any], 
        candidate_data: Dict[str, Any],
        include_ai_analysis: bool = True
    ) -> Dict[str, Any]:
        """Calculate detailed match score between specific job and candidate"""
        
        try:
            # Generate embeddings
            job_embedding_task = self._get_or_generate_job_embedding(job_data)
            candidate_embedding_task = self._get_or_generate_candidate_embedding(candidate_data)
            
            job_embedding, candidate_embedding = await asyncio.gather(
                job_embedding_task, candidate_embedding_task
            )
            
            # Calculate base match score
            match_result = await self.matching_model.calculate_match_score(
                job_data, candidate_data, job_embedding, candidate_embedding
            )
            
            # Add NLP analysis
            if self.nlp_service:
                skill_comparison = await self.nlp_service.compare_skill_sets(
                    job_data.get("skills", []),
                    candidate_data.get("skills", [])
                )
                match_result["skill_analysis"] = skill_comparison
                
                # Skill improvement suggestions
                suggestions = await self.nlp_service.suggest_skill_improvements(
                    job_data.get("skills", []),
                    candidate_data.get("skills", [])
                )
                match_result["improvement_suggestions"] = suggestions
            
            # Add AI-powered analysis
            if include_ai_analysis and self.openai_service:
                try:
                    ai_recommendations = await self.openai_service.generate_candidate_recommendations(
                        job_data, candidate_data, match_result["overall_score"]
                    )
                    match_result["ai_analysis"] = ai_recommendations
                except Exception as e:
                    logger.error(f"Failed to generate AI analysis: {e}")
                    match_result["ai_analysis"] = {"error": str(e)}
            
            # Add metadata
            match_result["analysis_metadata"] = {
                "job_id": job_data.get("id"),
                "candidate_id": candidate_data.get("id"),
                "calculated_at": datetime.now().isoformat(),
                "embedding_model": "sentence-transformers",
                "matching_model_version": "1.0.0"
            }
            
            return match_result
            
        except Exception as e:
            logger.error(f"Error calculating single match: {e}")
            return {
                "overall_score": 0,
                "confidence": 0,
                "error": str(e),
                "breakdown": {},
                "explanation": "Error calculating match score"
            }
    
    async def batch_match_jobs_candidates(
        self, 
        job_ids: List[str], 
        candidate_ids: List[str],
        min_score: float = 50.0
    ) -> Dict[str, Any]:
        """
        Perform batch matching between multiple jobs and candidates
        Optimized for processing large datasets
        """
        try:
            logger.info(f"Batch matching {len(job_ids)} jobs with {len(candidate_ids)} candidates")
            
            # This would typically fetch data from database
            # For now, we'll return a structure showing how it would work
            
            matches = []
            total_combinations = len(job_ids) * len(candidate_ids)
            processed = 0
            
            # Process in batches to avoid memory issues
            batch_size = 100
            
            for i in range(0, len(job_ids), batch_size):
                job_batch = job_ids[i:i + batch_size]
                
                # Get job embeddings in batch
                job_embeddings = {}
                for job_id in job_batch:
                    # Would fetch job data and embedding
                    job_embeddings[job_id] = {"embedding": [], "data": {}}
                
                for j in range(0, len(candidate_ids), batch_size):
                    candidate_batch = candidate_ids[j:j + batch_size]
                    
                    # Get candidate embeddings in batch
                    candidate_embeddings = {}
                    for candidate_id in candidate_batch:
                        # Would fetch candidate data and embedding
                        candidate_embeddings[candidate_id] = {"embedding": [], "data": {}}
                    
                    # Calculate matches for this batch
                    for job_id in job_batch:
                        for candidate_id in candidate_batch:
                            # Calculate match score
                            # match_score = await self.calculate_single_match(...)
                            
                            # Placeholder for actual calculation
                            match_score = {
                                "job_id": job_id,
                                "candidate_id": candidate_id,
                                "overall_score": 75.0,  # Placeholder
                                "calculated_at": datetime.now().isoformat()
                            }
                            
                            if match_score["overall_score"] >= min_score:
                                matches.append(match_score)
                            
                            processed += 1
                
                # Log progress
                progress = (processed / total_combinations) * 100
                logger.info(f"Batch matching progress: {progress:.1f}%")
            
            # Sort matches by score
            matches.sort(key=lambda x: x["overall_score"], reverse=True)
            
            return {
                "total_combinations": total_combinations,
                "matches_found": len(matches),
                "min_score_threshold": min_score,
                "processing_stats": {
                    "total_processed": processed,
                    "completion_rate": 100.0,
                    "processing_time": "calculated in real implementation"
                },
                "matches": matches[:1000]  # Limit to top 1000 matches
            }
            
        except Exception as e:
            logger.error(f"Error in batch matching: {e}")
            return {"error": str(e), "matches": []}
    
    async def _get_or_generate_job_embedding(self, job_data: Dict[str, Any]) -> List[float]:
        """Get job embedding from cache or generate new one"""
        
        job_id = job_data.get("id", "")
        
        # Check cache first
        if job_id in self._embedding_cache:
            return self._embedding_cache[job_id]
        
        # Check vector database
        if self.vector_service and job_id:
            result = await self.vector_service.get_job_embedding(job_id)
            if result:
                embedding, _ = result
                self._embedding_cache[job_id] = embedding
                return embedding
        
        # Generate new embedding
        if self.embedding_model:
            embedding = await self.embedding_model.embed_job_data(job_data)
            
            # Cache it
            if job_id:
                self._embedding_cache[job_id] = embedding
                
                # Store in vector database
                if self.vector_service:
                    await self.vector_service.store_job_embedding(
                        job_id, embedding, job_data
                    )
            
            return embedding
        
        # Fallback
        return [0.0] * 384
    
    async def _get_or_generate_candidate_embedding(self, candidate_data: Dict[str, Any]) -> List[float]:
        """Get candidate embedding from cache or generate new one"""
        
        candidate_id = candidate_data.get("id", "")
        
        # Check cache first
        if candidate_id in self._embedding_cache:
            return self._embedding_cache[candidate_id]
        
        # Check vector database
        if self.vector_service and candidate_id:
            result = await self.vector_service.get_candidate_embedding(candidate_id)
            if result:
                embedding, _ = result
                self._embedding_cache[candidate_id] = embedding
                return embedding
        
        # Generate new embedding
        if self.embedding_model:
            embedding = await self.embedding_model.embed_candidate_data(candidate_data)
            
            # Cache it
            if candidate_id:
                self._embedding_cache[candidate_id] = embedding
                
                # Store in vector database
                if self.vector_service:
                    await self.vector_service.store_candidate_embedding(
                        candidate_id, embedding, candidate_data
                    )
            
            return embedding
        
        # Fallback
        return [0.0] * 384
    
    async def _vector_search_candidates(
        self, 
        job_embedding: List[float], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search for similar candidates using vector similarity"""
        
        if not self.vector_service:
            return []
        
        return await self.vector_service.search_similar_candidates(
            job_embedding, limit, score_threshold=0.5
        )
    
    async def _vector_search_jobs(
        self, 
        candidate_embedding: List[float], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search for similar jobs using vector similarity"""
        
        if not self.vector_service:
            return []
        
        return await self.vector_service.search_similar_jobs(
            candidate_embedding, limit, score_threshold=0.5
        )
    
    async def _calculate_detailed_matches(
        self, 
        job_data: Dict[str, Any], 
        vector_candidates: List[Dict[str, Any]],
        job_embedding: List[float]
    ) -> List[Dict[str, Any]]:
        """Calculate detailed match scores for vector search results"""
        
        detailed_matches = []
        
        for candidate_result in vector_candidates:
            candidate_data = candidate_result.get("metadata", {})
            candidate_id = candidate_result.get("candidate_id")
            
            # Get candidate embedding
            candidate_embedding = await self._get_or_generate_candidate_embedding(candidate_data)
            
            # Calculate detailed match
            if self.matching_model:
                match_result = await self.matching_model.calculate_match_score(
                    job_data, candidate_data, job_embedding, candidate_embedding
                )
                
                match_result["candidate_id"] = candidate_id
                match_result["candidate_data"] = candidate_data
                match_result["vector_similarity"] = candidate_result.get("similarity_score", 0)
                
                detailed_matches.append(match_result)
        
        return detailed_matches
    
    async def _calculate_detailed_job_matches(
        self, 
        candidate_data: Dict[str, Any], 
        vector_jobs: List[Dict[str, Any]],
        candidate_embedding: List[float]
    ) -> List[Dict[str, Any]]:
        """Calculate detailed match scores for job vector search results"""
        
        detailed_matches = []
        
        for job_result in vector_jobs:
            job_data = job_result.get("metadata", {})
            job_id = job_result.get("job_id")
            
            # Get job embedding
            job_embedding = await self._get_or_generate_job_embedding(job_data)
            
            # Calculate detailed match
            if self.matching_model:
                match_result = await self.matching_model.calculate_match_score(
                    job_data, candidate_data, job_embedding, candidate_embedding
                )
                
                match_result["job_id"] = job_id
                match_result["job_data"] = job_data
                match_result["vector_similarity"] = job_result.get("similarity_score", 0)
                
                detailed_matches.append(match_result)
        
        return detailed_matches
    
    def clear_cache(self):
        """Clear all caches"""
        self._embedding_cache.clear()
        self._match_cache.clear()
        logger.info("Matching service caches cleared")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "embedding_cache_size": len(self._embedding_cache),
            "match_cache_size": len(self._match_cache),
            "services_initialized": {
                "matching_model": self.matching_model is not None,
                "embedding_model": self.embedding_model is not None,
                "vector_service": self.vector_service is not None,
                "nlp_service": self.nlp_service is not None,
                "openai_service": self.openai_service is not None
            }
        }
