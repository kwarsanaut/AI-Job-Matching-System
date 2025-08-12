"""
OpenAI Service for advanced AI operations
Handles OpenAI API integration for enhanced analysis and generation
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import openai
from openai import AsyncOpenAI
import tiktoken

from utils.config import get_settings

logger = logging.getLogger(__name__)

class OpenAIService:
    """
    Service for OpenAI API integration
    Provides advanced AI capabilities for job matching system
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[AsyncOpenAI] = None
        self.model_config = {
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.0015,
                "best_for": ["analysis", "extraction", "summarization"]
            },
            "gpt-4": {
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.03,
                "best_for": ["complex_analysis", "creative_writing", "detailed_recommendations"]
            },
            "text-embedding-ada-002": {
                "dimensions": 1536,
                "cost_per_1k_tokens": 0.0001,
                "best_for": ["embeddings", "similarity"]
            }
        }
        
        # Rate limiting
        self.request_count = 0
        self.last_reset = datetime.now()
        self.rate_limit = 100  # requests per minute
        
        # Token encoding
        self.encoding = None
    
    async def initialize(self):
        """Initialize OpenAI service"""
        try:
            if not self.settings.openai_api_key:
                logger.warning("OpenAI API key not provided. Service will be limited.")
                return
            
            logger.info("Initializing OpenAI service...")
            
            # Initialize client
            self.client = AsyncOpenAI(
                api_key=self.settings.openai_api_key
            )
            
            # Initialize token encoder
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
            # Test connection
            await self._test_connection()
            
            logger.info("✅ OpenAI service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI service: {e}")
            # Don't raise - service should work without OpenAI
    
    async def _test_connection(self):
        """Test OpenAI API connection"""
        try:
            if not self.client:
                return
            
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=5
            )
            
            logger.info("OpenAI API connection test successful")
            
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            raise e
    
    async def analyze_job_requirements_advanced(self, job_description: str) -> Dict[str, Any]:
        """Advanced job requirements analysis using GPT"""
        
        if not self.client:
            return {"error": "OpenAI client not available"}
        
        try:
            prompt = self._build_job_analysis_prompt(job_description)
            
            response = await self._make_chat_request(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-3.5-turbo",
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse JSON response
            analysis = json.loads(response)
            
            # Add metadata
            analysis["analysis_timestamp"] = datetime.now().isoformat()
            analysis["model_used"] = "gpt-3.5-turbo"
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            return {"error": "Failed to parse AI response"}
        except Exception as e:
            logger.error(f"Error in advanced job analysis: {e}")
            return {"error": str(e)}
    
    def _build_job_analysis_prompt(self, job_description: str) -> str:
        """Build prompt for job analysis"""
        
        return f"""
        Analyze this job description and extract structured information. Respond with valid JSON only.

        Job Description:
        {job_description}

        Extract the following information in JSON format:
        {{
            "required_skills": ["list of must-have technical skills"],
            "preferred_skills": ["list of nice-to-have skills"],
            "soft_skills": ["communication", "leadership", etc.],
            "experience_level": "entry|junior|mid|senior|principal",
            "experience_years": "estimated years required",
            "education_requirements": ["degree requirements"],
            "key_responsibilities": ["main job duties"],
            "company_culture_indicators": ["remote-friendly", "fast-paced", etc.],
            "salary_indicators": "any salary information mentioned",
            "growth_opportunities": ["career advancement mentions"],
            "industry_focus": "primary industry/domain",
            "work_arrangement": "remote|hybrid|onsite|flexible",
            "team_structure": "team size and structure indicators",
            "technologies_mentioned": ["specific tools, platforms, frameworks"],
            "urgency_level": "low|medium|high",
            "job_complexity": "routine|moderate|complex|highly_complex"
        }}
        """
    
    async def generate_candidate_recommendations(
        self, 
        job_data: Dict[str, Any], 
        candidate_data: Dict[str, Any],
        match_score: float
    ) -> Dict[str, Any]:
        """Generate personalized recommendations for candidate"""
        
        if not self.client:
            return {"error": "OpenAI client not available"}
        
        try:
            prompt = self._build_recommendation_prompt(job_data, candidate_data, match_score)
            
            response = await self._make_chat_request(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-3.5-turbo",
                max_tokens=800,
                temperature=0.4
            )
            
            recommendations = json.loads(response)
            recommendations["generated_at"] = datetime.now().isoformat()
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {"error": str(e)}
    
    def _build_recommendation_prompt(
        self, 
        job_data: Dict[str, Any], 
        candidate_data: Dict[str, Any], 
        match_score: float
    ) -> str:
        """Build prompt for generating recommendations"""
        
        return f"""
        Generate personalized recommendations for this job-candidate match. Respond with valid JSON only.

        Job Information:
        - Title: {job_data.get('title', 'N/A')}
        - Skills Required: {', '.join(job_data.get('skills', []))}
        - Experience Level: {job_data.get('experience_level', 'N/A')}

        Candidate Information:
        - Current Title: {candidate_data.get('title', 'N/A')}
        - Skills: {', '.join(candidate_data.get('skills', []))}
        - Experience: {candidate_data.get('experience', 'N/A')}

        Match Score: {match_score}/100

        Generate recommendations in JSON format:
        {{
            "overall_assessment": "brief overall assessment",
            "strengths": ["candidate's main strengths for this role"],
            "areas_for_improvement": ["specific skills or areas to develop"],
            "interview_preparation": ["key topics to prepare for interview"],
            "skill_development_plan": [
                {{
                    "skill": "skill name",
                    "priority": "high|medium|low",
                    "timeline": "suggested learning timeline",
                    "resources": "learning suggestions"
                }}
            ],
            "application_strategy": "advice on how to apply/position themselves",
            "salary_negotiation_tips": "relevant salary advice",
            "next_steps": ["immediate action items"]
        }}
        """
    
    async def generate_job_posting_improvements(self, job_description: str) -> Dict[str, Any]:
        """Suggest improvements for job posting"""
        
        if not self.client:
            return {"error": "OpenAI client not available"}
        
        try:
            prompt = f"""
            Analyze this job posting and suggest improvements to make it more attractive to candidates and improve hiring success. Respond with valid JSON only.

            Job Posting:
            {job_description}

            Provide suggestions in JSON format:
            {{
                "clarity_score": "1-10 rating of how clear the posting is",
                "attractiveness_score": "1-10 rating of how attractive it is to candidates",
                "improvements": [
                    {{
                        "category": "title|description|requirements|benefits|culture",
                        "issue": "what needs improvement",
                        "suggestion": "specific improvement suggestion",
                        "priority": "high|medium|low"
                    }}
                ],
                "missing_elements": ["important elements that should be added"],
                "tone_assessment": "professional|casual|technical|overly_formal",
                "recommended_tone": "suggested tone for target audience",
                "keyword_optimization": ["keywords to include for better visibility"],
                "estimated_response_rate": "predicted candidate response rate improvement"
            }}
            """
            
            response = await self._make_chat_request(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-3.5-turbo",
                max_tokens=1000,
                temperature=0.3
            )
            
            improvements = json.loads(response)
            improvements["analysis_date"] = datetime.now().isoformat()
            
            return improvements
            
        except Exception as e:
            logger.error(f"Error generating job posting improvements: {e}")
            return {"error": str(e)}
    
    async def create_interview_questions(
        self, 
        job_data: Dict[str, Any], 
        candidate_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate tailored interview questions"""
        
        if not self.client:
            return {"error": "OpenAI client not available"}
        
        try:
            prompt = f"""
            Generate tailored interview questions for this job-candidate combination. Respond with valid JSON only.

            Job: {job_data.get('title', 'N/A')}
            Required Skills: {', '.join(job_data.get('skills', []))}
            
            Candidate Background: {candidate_data.get('title', 'N/A')}
            Candidate Skills: {', '.join(candidate_data.get('skills', []))}

            Generate questions in JSON format:
            {{
                "technical_questions": [
                    {{
                        "question": "technical question",
                        "focus_area": "specific skill/technology",
                        "difficulty": "beginner|intermediate|advanced",
                        "expected_answer_points": ["key points to look for"]
                    }}
                ],
                "behavioral_questions": [
                    {{
                        "question": "behavioral question",
                        "purpose": "what this question assesses",
                        "good_answer_indicators": ["signs of a strong answer"]
                    }}
                ],
                "situational_questions": [
                    {{
                        "scenario": "work situation description",
                        "question": "how would you handle this",
                        "assessment_criteria": ["what to evaluate"]
                    }}
                ],
                "culture_fit_questions": ["questions to assess culture fit"],
                "questions_for_candidate": ["questions candidate should ask interviewer"]
            }}
            """
            
            response = await self._make_chat_request(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-3.5-turbo",
                max_tokens=1200,
                temperature=0.4
            )
            
            questions = json.loads(response)
            questions["generated_for"] = {
                "job_title": job_data.get('title'),
                "candidate_title": candidate_data.get('title'),
                "generation_date": datetime.now().isoformat()
            }
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating interview questions: {e}")
            return {"error": str(e)}
    
    async def analyze_market_trends(self, skills: List[str], industry: str = "") -> Dict[str, Any]:
        """Analyze market trends for given skills"""
        
        if not self.client:
            return {"error": "OpenAI client not available"}
        
        try:
            skills_text = ", ".join(skills)
            industry_context = f" in the {industry} industry" if industry else ""
            
            prompt = f"""
            Analyze current market trends for these skills{industry_context}. Respond with valid JSON only.

            Skills: {skills_text}
            Industry: {industry or "General Tech"}

            Provide analysis in JSON format:
            {{
                "skill_demand_analysis": [
                    {{
                        "skill": "skill name",
                        "demand_level": "low|medium|high|very_high",
                        "trend": "declining|stable|growing|rapidly_growing",
                        "market_context": "brief explanation of demand"
                    }}
                ],
                "emerging_technologies": ["related emerging technologies to consider"],
                "market_insights": {{
                    "high_demand_areas": ["areas with highest demand"],
                    "competitive_skills": ["skills with high competition"],
                    "salary_trends": "general salary trend information",
                    "geographic_hotspots": ["locations with high demand"]
                }},
                "recommendations": {{
                    "skills_to_develop": ["skills to add to stay competitive"],
                    "skills_to_prioritize": ["existing skills to focus on"],
                    "market_positioning": "advice on market positioning"
                }},
                "future_outlook": "6-month to 2-year outlook for these skills"
            }}
            """
            
            response = await self._make_chat_request(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-3.5-turbo",
                max_tokens=1000,
                temperature=0.3
            )
            
            analysis = json.loads(response)
            analysis["analysis_date"] = datetime.now().isoformat()
            analysis["skills_analyzed"] = skills
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {e}")
            return {"error": str(e)}
    
    async def _make_chat_request(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> str:
        """Make chat completion request with rate limiting"""
        
        # Check rate limiting
        await self._check_rate_limit()
        
        if not self.client:
            raise Exception("OpenAI client not available")
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            self.request_count += 1
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise e
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        
        current_time = datetime.now()
        
        # Reset counter every minute
        if (current_time - self.last_reset).seconds >= 60:
            self.request_count = 0
            self.last_reset = current_time
        
        # If we've hit the rate limit, wait
        if self.request_count >= self.rate_limit:
            wait_time = 60 - (current_time - self.last_reset).seconds
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.last_reset = datetime.now()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        
        if not self.encoding:
            return len(text.split())  # Rough estimate
        
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, text: str, model: str = "gpt-3.5-turbo") -> float:
        """Estimate cost for processing text"""
        
        if model not in self.model_config:
            return 0.0
        
        tokens = self.count_tokens(text)
        cost_per_1k = self.model_config[model]["cost_per_1k_tokens"]
        
        return (tokens / 1000) * cost_per_1k
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service usage statistics"""
        
        return {
            "requests_this_minute": self.request_count,
            "rate_limit": self.rate_limit,
            "available_models": list(self.model_config.keys()),
            "client_available": self.client is not None,
            "last_reset": self.last_reset.isoformat() if self.last_reset else None
        }
