"""
Advanced Matching Model for Job-Candidate Compatibility
Implements sophisticated ML-based matching algorithms
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

from utils.config import get_settings

logger = logging.getLogger(__name__)

class MatchingModel:
    """
    Advanced matching model that combines multiple factors:
    - Semantic similarity (embeddings)
    - Skill overlap
    - Experience level matching
    - Location compatibility
    - Salary alignment
    - Cultural fit indicators
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Matching weights (configurable)
        self.weights = {
            "semantic_similarity": 0.35,
            "skill_overlap": 0.25,
            "experience_match": 0.15,
            "location_compatibility": 0.10,
            "salary_alignment": 0.10,
            "cultural_fit": 0.05
        }
        
        # Experience level mappings
        self.experience_levels = {
            "entry": (0, 2),
            "junior": (1, 3),
            "mid": (3, 7),
            "senior": (7, 12),
            "lead": (10, 20),
            "principal": (12, 25)
        }
        
        # European countries for location matching
        self.eu_countries = {
            "germany", "netherlands", "france", "spain", "italy", 
            "poland", "belgium", "austria", "sweden", "denmark",
            "finland", "portugal", "czech republic", "ireland"
        }
    
    async def calculate_match_score(
        self, 
        job_data: Dict[str, Any], 
        candidate_data: Dict[str, Any],
        job_embedding: Optional[List[float]] = None,
        candidate_embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive match score between job and candidate
        
        Returns:
            Dictionary with overall score, breakdown, and explanation
        """
        try:
            scores = {}
            
            # 1. Semantic similarity (if embeddings provided)
            if job_embedding and candidate_embedding:
                scores["semantic_similarity"] = self._calculate_semantic_similarity(
                    job_embedding, candidate_embedding
                )
            else:
                scores["semantic_similarity"] = await self._calculate_text_similarity(
                    job_data, candidate_data
                )
            
            # 2. Skill overlap
            scores["skill_overlap"] = self._calculate_skill_overlap(
                job_data.get("skills", []),
                candidate_data.get("skills", [])
            )
            
            # 3. Experience match
            scores["experience_match"] = self._calculate_experience_match(
                job_data.get("description", ""),
                candidate_data.get("experience", ""),
                job_data.get("title", "")
            )
            
            # 4. Location compatibility
            scores["location_compatibility"] = self._calculate_location_compatibility(
                job_data.get("location", ""),
                candidate_data.get("location", ""),
                job_data.get("remote_allowed", False)
            )
            
            # 5. Salary alignment
            scores["salary_alignment"] = self._calculate_salary_alignment(
                job_data.get("salary_min", 0),
                job_data.get("salary_max", 0),
                candidate_data.get("salary_expectation_min", 0),
                candidate_data.get("salary_expectation_max", 0)
            )
            
            # 6. Cultural fit (basic implementation)
            scores["cultural_fit"] = self._calculate_cultural_fit(
                job_data, candidate_data
            )
            
            # Calculate weighted overall score
            overall_score = sum(
                scores[factor] * weight 
                for factor, weight in self.weights.items()
                if factor in scores
            )
            
            # Generate explanation
            explanation = self._generate_explanation(scores, overall_score)
            
            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(job_data, candidate_data)
            
            return {
                "overall_score": min(overall_score * 100, 100),  # Convert to percentage
                "confidence": confidence,
                "breakdown": {k: v * 100 for k, v in scores.items()},
                "explanation": explanation,
                "matched_skills": self._get_matched_skills(
                    job_data.get("skills", []),
                    candidate_data.get("skills", [])
                ),
                "recommendations": self._generate_recommendations(scores, job_data, candidate_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating match score: {e}")
            return {
                "overall_score": 0,
                "confidence": 0,
                "breakdown": {},
                "explanation": "Error calculating match score",
                "matched_skills": [],
                "recommendations": []
            }
    
    def _calculate_semantic_similarity(
        self, 
        job_embedding: List[float], 
        candidate_embedding: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        
        try:
            vec1 = np.array(job_embedding)
            vec2 = np.array(candidate_embedding)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0, min(1, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    async def _calculate_text_similarity(
        self, 
        job_data: Dict[str, Any], 
        candidate_data: Dict[str, Any]
    ) -> float:
        """Calculate text similarity using TF-IDF as fallback"""
        
        try:
            # Combine job text
            job_text_parts = [
                job_data.get("title", ""),
                job_data.get("description", ""),
                " ".join(job_data.get("skills", [])),
                " ".join(job_data.get("requirements", []))
            ]
            job_text = " ".join(filter(None, job_text_parts))
            
            # Combine candidate text
            candidate_text_parts = [
                candidate_data.get("title", ""),
                candidate_data.get("summary", ""),
                " ".join(candidate_data.get("skills", [])),
                candidate_data.get("experience", "")
            ]
            candidate_text = " ".join(filter(None, candidate_text_parts))
            
            if not job_text or not candidate_text:
                return 0.0
            
            # Calculate TF-IDF similarity
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([job_text, candidate_text])
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return max(0, min(1, similarity_matrix[0, 1]))
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _calculate_skill_overlap(
        self, 
        job_skills: List[str], 
        candidate_skills: List[str]
    ) -> float:
        """Calculate skill overlap score"""
        
        if not job_skills:
            return 1.0  # No requirements = perfect match
        
        if not candidate_skills:
            return 0.0
        
        # Normalize skills (lowercase)
        job_skills_normalized = [skill.lower().strip() for skill in job_skills]
        candidate_skills_normalized = [skill.lower().strip() for skill in candidate_skills]
        
        # Calculate exact matches
        exact_matches = len(set(job_skills_normalized) & set(candidate_skills_normalized))
        
        # Calculate partial matches (contains)
        partial_matches = 0
        for job_skill in job_skills_normalized:
            for candidate_skill in candidate_skills_normalized:
                if job_skill in candidate_skill or candidate_skill in job_skill:
                    if job_skill not in set(job_skills_normalized) & set(candidate_skills_normalized):
                        partial_matches += 0.5
                        break
        
        total_matches = exact_matches + partial_matches
        max_possible = len(job_skills_normalized)
        
        return min(1.0, total_matches / max_possible)
    
    def _calculate_experience_match(
        self, 
        job_description: str, 
        candidate_experience: str,
        job_title: str
    ) -> float:
        """Calculate experience level match"""
        
        try:
            # Extract required experience from job
            required_level = self._extract_experience_level(job_description, job_title)
            
            # Extract candidate experience years
            candidate_years = self._parse_experience_years(candidate_experience)
            
            # Get experience range for required level
            if required_level not in self.experience_levels:
                return 0.5  # Default if can't determine
            
            min_exp, max_exp = self.experience_levels[required_level]
            
            # Calculate match score
            if min_exp <= candidate_years <= max_exp:
                return 1.0  # Perfect match
            elif candidate_years < min_exp:
                # Underqualified
                gap = min_exp - candidate_years
                return max(0, 1 - (gap * 0.2))  # Penalty for each year short
            else:
                # Overqualified
                excess = candidate_years - max_exp
                return max(0.7, 1 - (excess * 0.05))  # Smaller penalty for overqualification
            
        except Exception as e:
            logger.error(f"Error calculating experience match: {e}")
            return 0.5
    
    def _extract_experience_level(self, job_description: str, job_title: str) -> str:
        """Extract required experience level from job description and title"""
        
        text = (job_description + " " + job_title).lower()
        
        # Priority order matters
        if any(term in text for term in ["principal", "staff", "distinguished"]):
            return "principal"
        elif any(term in text for term in ["lead", "team lead", "tech lead"]):
            return "lead"
        elif any(term in text for term in ["senior", "sr.", "10+", "8+", "7+"]):
            return "senior"
        elif any(term in text for term in ["mid-level", "mid level", "3-5", "4-6", "5-7"]):
            return "mid"
        elif any(term in text for term in ["junior", "jr.", "1-3", "2-4"]):
            return "junior"
        elif any(term in text for term in ["entry", "graduate", "intern", "0-2"]):
            return "entry"
        else:
            return "mid"  # Default assumption
    
    def _parse_experience_years(self, experience_str: str) -> int:
        """Parse experience years from candidate data"""
        
        import re
        
        if not experience_str:
            return 0
        
        # Look for patterns like "5 years", "3-5 years", etc.
        years_pattern = r'(\d+)(?:\s*-\s*\d+)?\s*(?:years?|yrs?)'
        matches = re.findall(years_pattern, experience_str.lower())
        
        if matches:
            return int(matches[0])
        
        # Look for just numbers
        number_pattern = r'\b(\d+)\b'
        numbers = re.findall(number_pattern, experience_str)
        
        if numbers:
            # Take the first reasonable number (between 0 and 50)
            for num in numbers:
                years = int(num)
                if 0 <= years <= 50:
                    return years
        
        return 0
    
    def _calculate_location_compatibility(
        self, 
        job_location: str, 
        candidate_location: str,
        remote_allowed: bool = False
    ) -> float:
        """Calculate location compatibility score"""
        
        if not job_location or not candidate_location:
            return 0.5  # Default if location info missing
        
        if remote_allowed:
            return 1.0  # Perfect if remote work allowed
        
        # Parse locations
        job_parts = [part.strip().lower() for part in job_location.split(',')]
        candidate_parts = [part.strip().lower() for part in candidate_location.split(',')]
        
        if not job_parts or not candidate_parts:
            return 0.5
        
        job_city = job_parts[0]
        job_country = job_parts[-1] if len(job_parts) > 1 else ""
        
        candidate_city = candidate_parts[0]
        candidate_country = candidate_parts[-1] if len(candidate_parts) > 1 else ""
        
        # Same city = perfect match
        if job_city == candidate_city:
            return 1.0
        
        # Same country = good match
        if job_country and candidate_country and job_country == candidate_country:
            return 0.8
        
        # Both in EU = decent match
        if (job_country in self.eu_countries and candidate_country in self.eu_countries):
            return 0.6
        
        # Different continents = poor match
        return 0.3
    
    def _calculate_salary_alignment(
        self,
        job_salary_min: int,
        job_salary_max: int,
        candidate_salary_min: int,
        candidate_salary_max: int
    ) -> float:
        """Calculate salary alignment score"""
        
        # If no salary info, return neutral score
        if not any([job_salary_min, job_salary_max, candidate_salary_min, candidate_salary_max]):
            return 0.5
        
        # Use job max and candidate min for comparison
        job_offer = job_salary_max or job_salary_min or 0
        candidate_expectation = candidate_salary_min or candidate_salary_max or 0
        
        if job_offer <= 0 or candidate_expectation <= 0:
            return 0.5
        
        # Calculate alignment
        ratio = job_offer / candidate_expectation
        
        if ratio >= 1.0:
            # Job pays at least what candidate expects
            return min(1.0, ratio / 1.2)  # Perfect if within 20% above expectation
        else:
            # Job pays less than expectation
            return max(0, ratio)  # Linear penalty
    
    def _calculate_cultural_fit(
        self,
        job_data: Dict[str, Any],
        candidate_data: Dict[str, Any]
    ) -> float:
        """Calculate cultural fit score (basic implementation)"""
        
        try:
            # Basic cultural fit indicators
            fit_score = 0.5  # Base score
            
            # Company size preference (if available)
            job_type = job_data.get("type", "").lower()
            candidate_availability = candidate_data.get("availability", "").lower()
            
            # Work type alignment
            if job_type and candidate_availability:
                if job_type == candidate_availability:
                    fit_score += 0.3
                elif "part" in job_type and "part" in candidate_availability:
                    fit_score += 0.2
                elif "full" in job_type and "full" in candidate_availability:
                    fit_score += 0.2
            
            # Industry experience (basic check)
            job_desc = job_data.get("description", "").lower()
            candidate_summary = candidate_data.get("summary", "").lower()
            
            # Look for industry keywords
            tech_keywords = ["startup", "scale-up", "enterprise", "corporate", "agency"]
            for keyword in tech_keywords:
                if keyword in job_desc and keyword in candidate_summary:
                    fit_score += 0.1
                    break
            
            return min(1.0, fit_score)
            
        except Exception as e:
            logger.error(f"Error calculating cultural fit: {e}")
            return 0.5
    
    def _generate_explanation(self, scores: Dict[str, float], overall_score: float) -> str:
        """Generate human-readable explanation"""
        
        explanations = []
        
        # Overall assessment
        if overall_score >= 0.8:
            explanations.append("Excellent overall match with strong alignment across multiple criteria")
        elif overall_score >= 0.6:
            explanations.append("Good match with solid compatibility in key areas")
        elif overall_score >= 0.4:
            explanations.append("Moderate match with some areas for development")
        else:
            explanations.append("Limited match - significant gaps in requirements")
        
        # Specific strengths
        strong_areas = [area for area, score in scores.items() if score >= 0.7]
        if strong_areas:
            areas_text = ", ".join(strong_areas).replace("_", " ")
            explanations.append(f"Strong in: {areas_text}")
        
        # Areas for improvement
        weak_areas = [area for area, score in scores.items() if score < 0.4]
        if weak_areas:
            areas_text = ", ".join(weak_areas).replace("_", " ")
            explanations.append(f"Development needed in: {areas_text}")
        
        return ". ".join(explanations) + "."
    
    def _get_matched_skills(self, job_skills: List[str], candidate_skills: List[str]) -> List[str]:
        """Get list of matched skills between job and candidate"""
        
        if not job_skills or not candidate_skills:
            return []
        
        job_skills_norm = [skill.lower().strip() for skill in job_skills]
        candidate_skills_norm = [skill.lower().strip() for skill in candidate_skills]
        
        matched = []
        
        # Exact matches
        for job_skill, orig_job_skill in zip(job_skills_norm, job_skills):
            if job_skill in candidate_skills_norm:
                matched.append(orig_job_skill)
        
        # Partial matches
        for job_skill, orig_job_skill in zip(job_skills_norm, job_skills):
            if orig_job_skill not in matched:
                for candidate_skill in candidate_skills_norm:
                    if job_skill in candidate_skill or candidate_skill in job_skill:
                        matched.append(orig_job_skill)
                        break
        
        return matched
    
    def _generate_recommendations(
        self, 
        scores: Dict[str, float], 
        job_data: Dict[str, Any], 
        candidate_data: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving the match"""
        
        recommendations = []
        
        # Skill gap recommendations
        if scores.get("skill_overlap", 0) < 0.6:
            job_skills = set(skill.lower() for skill in job_data.get("skills", []))
            candidate_skills = set(skill.lower() for skill in candidate_data.get("skills", []))
            missing_skills = job_skills - candidate_skills
            
            if missing_skills and len(missing_skills) <= 3:
                skills_text = ", ".join(list(missing_skills)[:3])
                recommendations.append(f"Consider training in: {skills_text}")
        
        # Experience recommendations
        if scores.get("experience_match", 0) < 0.5:
            recommendations.append("Experience level may not align perfectly - consider role level adjustment")
        
        # Location recommendations  
        if scores.get("location_compatibility", 0) < 0.6:
            if not job_data.get("remote_allowed", False):
                recommendations.append("Consider remote work options to improve location compatibility")
        
        # Salary recommendations
        if scores.get("salary_alignment", 0) < 0.5:
            recommendations.append("Salary expectations may need adjustment for mutual alignment")
        
        return recommendations
    
    def _calculate_confidence(self, job_data: Dict[str, Any], candidate_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on data completeness"""
        
        job_completeness = 0
        candidate_completeness = 0
        
        # Job data completeness (out of 100)
        if job_data.get("title"): job_completeness += 20
        if job_data.get("description"): job_completeness += 25
        if job_data.get("skills"): job_completeness += 25
        if job_data.get("location"): job_completeness += 15
        if job_data.get("salary_min") or job_data.get("salary_max"): job_completeness += 15
        
        # Candidate data completeness (out of 100)
        if candidate_data.get("title"): candidate_completeness += 20
        if candidate_data.get("summary"): candidate_completeness += 25
        if candidate_data.get("skills"): candidate_completeness += 25
        if candidate_data.get("experience"): candidate_completeness += 15
        if candidate_data.get("salary_expectation_min") or candidate_data.get("salary_expectation_max"): candidate_completeness += 15
        
        # Overall confidence is average of both completeness scores
        overall_completeness = (job_completeness + candidate_completeness) / 200
        return min(1.0, overall_completeness)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics and configuration"""
        
        return {
            "weights": self.weights,
            "experience_levels": self.experience_levels,
            "supported_countries": len(self.eu_countries),
            "model_version": "1.0.0",
            "last_updated": datetime.now().isoformat()
        }
