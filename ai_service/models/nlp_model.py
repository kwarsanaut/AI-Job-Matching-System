"""
NLP Model for text analysis and skill extraction
Handles natural language processing tasks for job matching
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from transformers import pipeline
import openai
from openai import AsyncOpenAI

from utils.config import get_settings

logger = logging.getLogger(__name__)

class NLPModel:
    """
    Advanced NLP model for job matching tasks:
    - Skill extraction from text
    - Job requirement analysis
    - Sentiment analysis
    - Entity recognition
    - Text classification
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.nlp = None
        self.sentiment_analyzer = None
        self.openai_client = None
        
        # Skill taxonomies and patterns
        self.tech_skills = {
            'programming_languages': [
                'python', 'javascript', 'java', 'c++', 'c#', 'golang', 'go', 'rust', 
                'typescript', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab',
                'perl', 'shell', 'bash', 'powershell'
            ],
            'frameworks': [
                'react', 'vue', 'angular', 'nodejs', 'express', 'django', 'flask',
                'spring', 'hibernate', 'laravel', 'rails', 'gin', 'echo', 'fiber',
                'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'pandas', 'numpy'
            ],
            'databases': [
                'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'dynamodb', 'sqlite', 'oracle', 'sql server', 'influxdb', 'neo4j',
                'qdrant', 'pinecone', 'weaviate', 'chroma'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google cloud', 'digitalocean', 'heroku',
                'vercel', 'netlify', 'firebase', 'cloudflare'
            ],
            'devops_tools': [
                'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions',
                'terraform', 'ansible', 'chef', 'puppet', 'vagrant', 'helm'
            ],
            'ai_ml': [
                'machine learning', 'deep learning', 'neural networks', 'nlp',
                'natural language processing', 'computer vision', 'rag',
                'retrieval augmented generation', 'llm', 'large language models',
                'transformers', 'bert', 'gpt', 'vector databases', 'embeddings'
            ]
        }
        
        # Soft skills patterns
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'analytical thinking', 'creativity', 'adaptability', 'time management',
            'project management', 'critical thinking', 'collaboration', 'mentoring'
        ]
        
        # Experience level patterns
        self.experience_patterns = {
            'entry': r'\b(?:entry|graduate|intern|junior|0-2|1-2)\b',
            'junior': r'\b(?:junior|jr\.?|1-3|2-3)\b',
            'mid': r'\b(?:mid-?level|3-5|4-6|5-7)\b',
            'senior': r'\b(?:senior|sr\.?|7\+|8\+|lead)\b',
            'principal': r'\b(?:principal|staff|distinguished|architect)\b'
        }
        
    async def initialize(self):
        """Initialize NLP models and tools"""
        try:
            logger.info("Initializing NLP models...")
            
            # Load spaCy model
            logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            
            # Download required NLTK data
            try:
                nltk.data.find('corpora/stopwords')
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('taggers/averaged_perceptron_tagger')
                nltk.data.find('chunkers/maxent_ne_chunker')
                nltk.data.find('corpora/words')
            except LookupError:
                logger.info("Downloading NLTK data...")
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)
            
            # Initialize sentiment analyzer
            logger.info("Loading sentiment analyzer...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Initialize OpenAI client
            if self.settings.openai_api_key:
                self.openai_client = AsyncOpenAI(
                    api_key=self.settings.openai_api_key
                )
            
            logger.info("✅ NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize NLP models: {e}")
            raise e
    
    async def extract_skills_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract technical and soft skills from text"""
        
        if not text:
            return {"technical_skills": [], "soft_skills": []}
        
        text_lower = text.lower()
        
        # Extract technical skills
        technical_skills = []
        for category, skills in self.tech_skills.items():
            for skill in skills:
                if skill in text_lower:
                    # Check if it's a whole word match or reasonable partial match
                    if self._is_valid_skill_match(skill, text_lower):
                        technical_skills.append(skill.title())
        
        # Extract soft skills
        soft_skills = []
        for skill in self.soft_skills:
            if skill in text_lower:
                if self._is_valid_skill_match(skill, text_lower):
                    soft_skills.append(skill.title())
        
        # Use spaCy for additional entity extraction
        if self.nlp:
            additional_skills = await self._extract_with_spacy(text)
            technical_skills.extend(additional_skills)
        
        # Remove duplicates and sort
        technical_skills = sorted(list(set(technical_skills)))
        soft_skills = sorted(list(set(soft_skills)))
        
        return {
            "technical_skills": technical_skills,
            "soft_skills": soft_skills
        }
    
    def _is_valid_skill_match(self, skill: str, text: str) -> bool:
        """Check if skill match is valid (not part of another word)"""
        
        # For single words, ensure word boundaries
        if ' ' not in skill:
            pattern = r'\b' + re.escape(skill) + r'\b'
            return bool(re.search(pattern, text, re.IGNORECASE))
        
        # For multi-word skills, simple contains check
        return skill in text
    
    async def _extract_with_spacy(self, text: str) -> List[str]:
        """Extract additional skills using spaCy NLP"""
        
        try:
            doc = self.nlp(text)
            
            skills = []
            
            # Look for technology-related entities
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT"] and len(ent.text) > 2:
                    # Check if it looks like a technology term
                    if any(tech_term in ent.text.lower() for tech_term in 
                          ['tech', 'soft', 'data', 'web', 'api', 'cloud', 'system']):
                        skills.append(ent.text)
            
            # Look for noun phrases that might be skills
            for chunk in doc.noun_chunks:
                if 2 <= len(chunk.text) <= 30:  # Reasonable length
                    chunk_lower = chunk.text.lower()
                    if any(keyword in chunk_lower for keyword in 
                          ['development', 'programming', 'analysis', 'design', 'management']):
                        skills.append(chunk.text.title())
            
            return skills[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Error in spaCy extraction: {e}")
            return []
    
    async def analyze_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """Analyze job description to extract structured requirements"""
        
        if not job_description:
            return {}
        
        try:
            # Extract skills
            skills_data = await self.extract_skills_from_text(job_description)
            
            # Extract experience level
            experience_level = self._extract_experience_level(job_description)
            
            # Extract education requirements
            education_req = self._extract_education_requirements(job_description)
            
            # Extract key responsibilities
            responsibilities = self._extract_responsibilities(job_description)
            
            # Extract nice-to-have vs must-have
            requirements_categorized = self._categorize_requirements(job_description)
            
            # Use OpenAI for enhanced analysis if available
            ai_analysis = {}
            if self.openai_client:
                ai_analysis = await self._analyze_with_openai(job_description)
            
            return {
                "technical_skills": skills_data["technical_skills"],
                "soft_skills": skills_data["soft_skills"],
                "experience_level": experience_level,
                "education_requirements": education_req,
                "key_responsibilities": responsibilities,
                "must_have": requirements_categorized["must_have"],
                "nice_to_have": requirements_categorized["nice_to_have"],
                "ai_insights": ai_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing job requirements: {e}")
            return {}
    
    def _extract_experience_level(self, text: str) -> str:
        """Extract required experience level from text"""
        
        text_lower = text.lower()
        
        # Check patterns in order of specificity
        for level, pattern in self.experience_patterns.items():
            if re.search(pattern, text_lower):
                return level
        
        # Look for year patterns
        year_pattern = r'(\d+)(?:\+)?\s*(?:years?|yrs?)'
        matches = re.findall(year_pattern, text_lower)
        
        if matches:
            years = int(matches[0])
            if years <= 2:
                return 'entry'
            elif years <= 4:
                return 'junior'
            elif years <= 7:
                return 'mid'
            elif years <= 12:
                return 'senior'
            else:
                return 'principal'
        
        return 'mid'  # Default
    
    def _extract_education_requirements(self, text: str) -> List[str]:
        """Extract education requirements from text"""
        
        text_lower = text.lower()
        requirements = []
        
        education_patterns = {
            'high_school': r'\b(?:high school|secondary school|diploma)\b',
            'bachelors': r'\b(?:bachelor|b\.?s\.?|b\.?a\.?|undergraduate)\b',
            'masters': r'\b(?:master|m\.?s\.?|m\.?a\.?|graduate)\b',
            'phd': r'\b(?:ph\.?d\.?|doctorate|doctoral)\b',
            'certification': r'\b(?:certification|certified|certificate)\b'
        }
        
        for edu_type, pattern in education_patterns.items():
            if re.search(pattern, text_lower):
                requirements.append(edu_type.replace('_', ' ').title())
        
        return requirements
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract key responsibilities from job description"""
        
        responsibilities = []
        
        # Look for responsibility sections
        responsibility_markers = [
            'responsibilities:', 'duties:', 'you will:', 'what you\'ll do:',
            'key responsibilities:', 'main duties:'
        ]
        
        text_lower = text.lower()
        
        # Find responsibility section
        start_idx = -1
        for marker in responsibility_markers:
            idx = text_lower.find(marker)
            if idx != -1:
                start_idx = idx + len(marker)
                break
        
        if start_idx == -1:
            # No explicit section, look for bullet points or action verbs
            sentences = sent_tokenize(text)
            for sentence in sentences[:10]:  # Check first 10 sentences
                if self._looks_like_responsibility(sentence):
                    responsibilities.append(sentence.strip())
        else:
            # Extract from responsibility section
            responsibility_text = text[start_idx:start_idx + 1000]  # Next 1000 chars
            lines = responsibility_text.split('\n')
            
            for line in lines[:10]:  # Max 10 responsibilities
                line = line.strip()
                if line and len(line) > 10:  # Meaningful content
                    # Clean bullet points
                    line = re.sub(r'^[-•*]\s*', '', line)
                    if line:
                        responsibilities.append(line)
        
        return responsibilities[:5]  # Return top 5
    
    def _looks_like_responsibility(self, sentence: str) -> bool:
        """Check if sentence looks like a responsibility description"""
        
        sentence_lower = sentence.lower()
        
        # Action verbs that indicate responsibilities
        action_verbs = [
            'develop', 'build', 'create', 'design', 'implement', 'maintain',
            'lead', 'manage', 'coordinate', 'collaborate', 'work with',
            'analyze', 'research', 'optimize', 'improve', 'support'
        ]
        
        return any(verb in sentence_lower for verb in action_verbs)
    
    def _categorize_requirements(self, text: str) -> Dict[str, List[str]]:
        """Categorize requirements into must-have vs nice-to-have"""
        
        text_lower = text.lower()
        
        must_have = []
        nice_to_have = []
        
        # Look for explicit sections
        nice_to_have_markers = [
            'nice to have:', 'preferred:', 'bonus:', 'plus:', 'advantageous:',
            'desired:', 'preferred qualifications:', 'nice-to-have:'
        ]
        
        required_markers = [
            'required:', 'must have:', 'essential:', 'mandatory:', 'requirements:',
            'qualifications:', 'you must:', 'required skills:'
        ]
        
        # Simple categorization based on keywords
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            if any(marker in sentence_lower for marker in nice_to_have_markers):
                nice_to_have.append(sentence.strip())
            elif any(marker in sentence_lower for marker in required_markers):
                must_have.append(sentence.strip())
            elif 'prefer' in sentence_lower or 'bonus' in sentence_lower:
                nice_to_have.append(sentence.strip())
            elif 'require' in sentence_lower or 'must' in sentence_lower:
                must_have.append(sentence.strip())
        
        return {
            "must_have": must_have[:5],
            "nice_to_have": nice_to_have[:5]
        }
    
    async def _analyze_with_openai(self, job_description: str) -> Dict[str, Any]:
        """Use OpenAI for enhanced job analysis"""
        
        try:
            prompt = f"""
            Analyze this job description and provide insights:
            
            {job_description}
            
            Please provide:
            1. Top 5 most important skills required
            2. Company culture indicators
            3. Growth opportunities mentioned
            4. Estimated salary range (in EUR)
            5. Remote work indicators
            
            Respond in JSON format.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse JSON response
            import json
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return {}
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        
        if not text or not self.sentiment_analyzer:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        try:
            # Truncate text if too long
            text = text[:512]  # Model limit
            
            results = self.sentiment_analyzer(text)
            
            # Get the highest confidence prediction
            best_result = max(results[0], key=lambda x: x['score'])
            
            return {
                "sentiment": best_result['label'].lower(),
                "confidence": best_result['score'],
                "all_scores": results[0]
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.0}
    
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        
        if not text or not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            
            # Clean up and deduplicate
            for label in entities:
                entities[label] = list(set(entities[label]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}
    
    def get_skill_categories(self) -> Dict[str, List[str]]:
        """Get all available skill categories"""
        return self.tech_skills
    
    async def validate_skills(self, skills: List[str]) -> Dict[str, bool]:
        """Validate if provided skills are recognized"""
        
        all_known_skills = []
        for category_skills in self.tech_skills.values():
            all_known_skills.extend([skill.lower() for skill in category_skills])
        all_known_skills.extend([skill.lower() for skill in self.soft_skills])
        
        validation = {}
        for skill in skills:
            validation[skill] = skill.lower() in all_known_skills
        
        return validation
