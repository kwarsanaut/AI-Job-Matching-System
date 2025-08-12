"""
NLP Service for high-level natural language processing tasks
Provides business logic layer for NLP operations
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio

from models.nlp_model import NLPModel
from utils.config import get_settings

logger = logging.getLogger(__name__)

class NLPService:
    """
    High-level service for NLP operations in job matching system
    Provides business logic and orchestration for NLP tasks
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.nlp_model: Optional[NLPModel] = None
        
        # Cache for processed results
        self._skill_cache = {}
        self._analysis_cache = {}
        
    async def initialize(self):
        """Initialize NLP service"""
        try:
            logger.info("Initializing NLP service...")
            
            self.nlp_model = NLPModel()
            await self.nlp_model.initialize()
            
            logger.info("✅ NLP service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize NLP service: {e}")
            raise e
    
    async def process_job_posting(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive processing of job posting
        Extracts skills, requirements, and analyzes content
        """
        try:
            job_text = self._combine_job_text(job_data)
            
            # Parallel processing of different analyses
            tasks = [
                self._extract_job_skills(job_text),
                self._analyze_job_requirements(job_text),
                self._analyze_sentiment(job_text),
                self._extract_entities(job_text)
            ]
            
            skills_data, requirements_data, sentiment_data, entities_data = await asyncio.gather(*tasks)
            
            # Combine results
            processed_data = {
                "job_id": job_data.get("id"),
                "skills": skills_data,
                "requirements": requirements_data,
                "sentiment": sentiment_data,
                "entities": entities_data,
                "metadata": {
                    "processed_at": asyncio.get_event_loop().time(),
                    "text_length": len(job_text),
                    "processing_version": "1.0.0"
                }
            }
            
            # Cache result
            if job_data.get("id"):
                self._analysis_cache[job_data["id"]] = processed_data
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing job posting: {e}")
            return {
                "error": str(e),
                "job_id": job_data.get("id"),
                "skills": {"technical_skills": [], "soft_skills": []},
                "requirements": {},
                "sentiment": {"sentiment": "neutral", "confidence": 0.0}
            }
    
    async def process_candidate_profile(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive processing of candidate profile
        Extracts skills, experience, and analyzes content
        """
        try:
            candidate_text = self._combine_candidate_text(candidate_data)
            
            # Parallel processing
            tasks = [
                self._extract_candidate_skills(candidate_text),
                self._analyze_candidate_experience(candidate_data),
                self._analyze_sentiment(candidate_text),
                self._extract_entities(candidate_text)
            ]
            
            skills_data, experience_data, sentiment_data, entities_data = await asyncio.gather(*tasks)
            
            # Combine results
            processed_data = {
                "candidate_id": candidate_data.get("id"),
                "skills": skills_data,
                "experience_analysis": experience_data,
                "sentiment": sentiment_data,
                "entities": entities_data,
                "metadata": {
                    "processed_at": asyncio.get_event_loop().time(),
                    "text_length": len(candidate_text),
                    "processing_version": "1.0.0"
                }
            }
            
            # Cache result
            if candidate_data.get("id"):
                self._analysis_cache[candidate_data["id"]] = processed_data
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing candidate profile: {e}")
            return {
                "error": str(e),
                "candidate_id": candidate_data.get("id"),
                "skills": {"technical_skills": [], "soft_skills": []},
                "experience_analysis": {},
                "sentiment": {"sentiment": "neutral", "confidence": 0.0}
            }
    
    def _combine_job_text(self, job_data: Dict[str, Any]) -> str:
        """Combine relevant job fields into single text"""
        
        text_parts = []
        
        if job_data.get("title"):
            text_parts.append(f"Job Title: {job_data['title']}")
        
        if job_data.get("description"):
            text_parts.append(f"Description: {job_data['description']}")
        
        if job_data.get("requirements"):
            if isinstance(job_data["requirements"], list):
                req_text = ". ".join(job_data["requirements"])
            else:
                req_text = str(job_data["requirements"])
            text_parts.append(f"Requirements: {req_text}")
        
        if job_data.get("skills"):
            skills_text = ", ".join(job_data["skills"])
            text_parts.append(f"Required Skills: {skills_text}")
        
        return ". ".join(text_parts)
    
    def _combine_candidate_text(self, candidate_data: Dict[str, Any]) -> str:
        """Combine relevant candidate fields into single text"""
        
        text_parts = []
        
        if candidate_data.get("title"):
            text_parts.append(f"Current Title: {candidate_data['title']}")
        
        if candidate_data.get("summary"):
            text_parts.append(f"Summary: {candidate_data['summary']}")
        
        if candidate_data.get("experience"):
            text_parts.append(f"Experience: {candidate_data['experience']}")
        
        if candidate_data.get("skills"):
            skills_text = ", ".join(candidate_data["skills"])
            text_parts.append(f"Skills: {skills_text}")
        
        if candidate_data.get("education"):
            text_parts.append(f"Education: {candidate_data['education']}")
        
        return ". ".join(text_parts)
    
    async def _extract_job_skills(self, job_text: str) -> Dict[str, List[str]]:
        """Extract skills from job text with caching"""
        
        # Check cache
        cache_key = f"job_skills_{hash(job_text)}"
        if cache_key in self._skill_cache:
            return self._skill_cache[cache_key]
        
        if not self.nlp_model:
            return {"technical_skills": [], "soft_skills": []}
        
        skills = await self.nlp_model.extract_skills_from_text(job_text)
        
        # Cache result
        self._skill_cache[cache_key] = skills
        
        return skills
    
    async def _extract_candidate_skills(self, candidate_text: str) -> Dict[str, List[str]]:
        """Extract skills from candidate text with caching"""
        
        # Check cache
        cache_key = f"candidate_skills_{hash(candidate_text)}"
        if cache_key in self._skill_cache:
            return self._skill_cache[cache_key]
        
        if not self.nlp_model:
            return {"technical_skills": [], "soft_skills": []}
        
        skills = await self.nlp_model.extract_skills_from_text(candidate_text)
        
        # Cache result
        self._skill_cache[cache_key] = skills
        
        return skills
    
    async def _analyze_job_requirements(self, job_text: str) -> Dict[str, Any]:
        """Analyze job requirements"""
        
        if not self.nlp_model:
            return {}
        
        return await self.nlp_model.analyze_job_requirements(job_text)
    
    async def _analyze_candidate_experience(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze candidate experience and background"""
        
        try:
            experience_analysis = {}
            
            # Parse experience years
            experience_str = candidate_data.get("experience", "")
            if experience_str:
                years = self._parse_experience_years(experience_str)
                experience_analysis["years"] = years
                experience_analysis["level"] = self._determine_experience_level(years)
            
            # Analyze education
            education = candidate_data.get("education", "")
            if education:
                experience_analysis["education_level"] = self._analyze_education_level(education)
            
            # Skills analysis
            skills = candidate_data.get("skills", [])
            if skills:
                experience_analysis["skill_diversity"] = len(set(skills))
                experience_analysis["technical_focus"] = self._analyze_technical_focus(skills)
            
            return experience_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing candidate experience: {e}")
            return {}
    
    def _parse_experience_years(self, experience_str: str) -> int:
        """Parse experience years from text"""
        
        import re
        
        if not experience_str:
            return 0
        
        # Look for patterns like "5 years", "3-5 years", etc.
        year_patterns = [
            r'(\d+)\s*(?:-\s*\d+)?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d{1,2})\s*(?:years?|yrs?)'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, experience_str.lower())
            if matches:
                return int(matches[0])
        
        return 0
    
    def _determine_experience_level(self, years: int) -> str:
        """Determine experience level from years"""
        
        if years <= 1:
            return "entry"
        elif years <= 3:
            return "junior"
        elif years <= 7:
            return "mid"
        elif years <= 12:
            return "senior"
        else:
            return "principal"
    
    def _analyze_education_level(self, education: str) -> str:
        """Analyze education level from text"""
        
        education_lower = education.lower()
        
        if any(term in education_lower for term in ["phd", "doctorate", "doctoral"]):
            return "doctorate"
        elif any(term in education_lower for term in ["master", "msc", "mba", "ms"]):
            return "masters"
        elif any(term in education_lower for term in ["bachelor", "bsc", "ba", "bs"]):
            return "bachelors"
        elif any(term in education_lower for term in ["diploma", "certificate"]):
            return "diploma"
        else:
            return "unknown"
    
    def _analyze_technical_focus(self, skills: List[str]) -> Dict[str, int]:
        """Analyze technical focus areas from skills"""
        
        if not self.nlp_model:
            return {}
        
        skill_categories = self.nlp_model.get_skill_categories()
        
        focus_areas = {}
        skills_lower = [skill.lower() for skill in skills]
        
        for category, category_skills in skill_categories.items():
            count = 0
            for skill in category_skills:
                if skill.lower() in skills_lower:
                    count += 1
            
            if count > 0:
                focus_areas[category] = count
        
        return focus_areas
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        
        if not self.nlp_model:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        return await self.nlp_model.analyze_sentiment(text)
    
    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        
        if not self.nlp_model:
            return {}
        
        return await self.nlp_model.extract_entities(text)
    
    async def compare_skill_sets(
        self, 
        job_skills: List[str], 
        candidate_skills: List[str]
    ) -> Dict[str, Any]:
        """Compare job requirements with candidate skills"""
        
        try:
            # Normalize skills
            job_skills_norm = [skill.lower().strip() for skill in job_skills]
            candidate_skills_norm = [skill.lower().strip() for skill in candidate_skills]
            
            # Find matches
            exact_matches = list(set(job_skills_norm) & set(candidate_skills_norm))
            
            # Find partial matches
            partial_matches = []
            for job_skill in job_skills_norm:
                if job_skill not in exact_matches:
                    for candidate_skill in candidate_skills_norm:
                        if (job_skill in candidate_skill or candidate_skill in job_skill) and len(job_skill) > 2:
                            partial_matches.append((job_skill, candidate_skill))
                            break
            
            # Find missing skills
            missing_skills = [skill for skill in job_skills_norm if skill not in exact_matches]
            
            # Calculate match percentage
            total_required = len(job_skills_norm)
            matched = len(exact_matches) + len(partial_matches) * 0.5
            match_percentage = (matched / total_required * 100) if total_required > 0 else 0
            
            return {
                "exact_matches": exact_matches,
                "partial_matches": partial_matches,
                "missing_skills": missing_skills,
                "match_percentage": round(match_percentage, 2),
                "total_required": total_required,
                "total_matched": len(exact_matches),
                "total_partial": len(partial_matches)
            }
            
        except Exception as e:
            logger.error(f"Error comparing skill sets: {e}")
            return {
                "exact_matches": [],
                "partial_matches": [],
                "missing_skills": job_skills,
                "match_percentage": 0,
                "error":
