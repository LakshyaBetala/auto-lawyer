import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from backend.local_llm.llm_runner import get_llm_runner
from backend.config import get_config

# Configure logging
logger = logging.getLogger(__name__)

class ResearchType(Enum):
    """Types of legal research that can be performed"""
    CASE_LAW = "case_law"
    STATUTORY = "statutory"
    REGULATORY = "regulatory"
    CONSTITUTIONAL = "constitutional"
    PRECEDENT = "precedent"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    GENERAL = "general"

class JurisdictionLevel(Enum):
    """Legal jurisdiction levels"""
    FEDERAL = "federal"
    STATE = "state"
    LOCAL = "local"
    INTERNATIONAL = "international"
    MIXED = "mixed"

@dataclass
class ResearchQuery:
    """Structured research query"""
    question: str
    research_type: ResearchType = ResearchType.GENERAL
    jurisdiction: JurisdictionLevel = JurisdictionLevel.FEDERAL
    focus_areas: List[str] = field(default_factory=list)
    time_period: Optional[str] = None
    priority_level: str = "medium"  # low, medium, high, critical
    context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "question": self.question,
            "research_type": self.research_type.value,
            "jurisdiction": self.jurisdiction.value,
            "focus_areas": self.focus_areas,
            "time_period": self.time_period,
            "priority_level": self.priority_level,
            "context": self.context
        }

@dataclass
class ResearchResult:
    """Research result with metadata"""
    query_id: str
    findings: str
    sources_cited: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    research_depth: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "query_id": self.query_id,
            "findings": self.findings,
            "sources_cited": self.sources_cited,
            "confidence_score": self.confidence_score,
            "research_depth": self.research_depth,
            "timestamp": self.timestamp.isoformat(),
            "warnings": self.warnings,
            "follow_up_suggestions": self.follow_up_suggestions
        }

class LegalResearcher:
    """
    Legal Research Agent powered by local LLM
    Conducts thorough legal research and analysis for brief drafting
    """
    
    def __init__(self):
        """Initialize the Legal Researcher agent"""
        self.config = get_config()
        self.llm = get_llm_runner()
        self.agent_name = "Legal Researcher"
        self.agent_id = "researcher"
        
        # Research configuration
        self.max_research_depth = self.config.legal.max_research_depth
        self.require_citations = self.config.legal.require_source_citations
        
        # Load agent-specific LLM settings
        self.llm_config = self._load_agent_config()
        
        logger.info(f"{self.agent_name} initialized successfully")
    
    def _load_agent_config(self) -> Dict[str, Any]:
        """Load researcher-specific LLM configuration"""
        try:
            # Get model config path from main config
            with open(self.config.llm.model_config_path, 'r') as f:
                model_config = json.load(f)
            
            # Extract researcher-specific overrides
            return model_config.get("agent_overrides", {}).get("researcher", {})
        except Exception as e:
            logger.warning(f"Could not load agent config: {e}. Using defaults.")
            return {
                "temperature": 0.1,
                "max_tokens": 3072,
                "top_p": 0.95
            }
    
    def _generate_query_id(self, query: ResearchQuery) -> str:
        """Generate unique ID for research query"""
        timestamp = int(time.time())
        query_hash = hash(query.question)
        return f"research_{timestamp}_{abs(query_hash)}"
    
    def _build_research_prompt(self, query: ResearchQuery) -> str:
        """Build comprehensive research prompt for the LLM"""
        
        # Get system prompt from config
        try:
            with open(self.config.llm.model_config_path, 'r') as f:
                model_config = json.load(f)
            system_prompt = model_config.get("system_prompts", {}).get("researcher", "")
        except:
            system_prompt = "<|im_start|>system\nYou are an expert legal research assistant.<|im_end|>"
        
        # Build detailed prompt
        prompt_parts = [
            system_prompt,
            "<|im_start|>user",
            f"Research Request: {query.question}",
            "",
            "Research Parameters:",
            f"- Research Type: {query.research_type.value.replace('_', ' ').title()}",
            f"- Jurisdiction: {query.jurisdiction.value.title()}",
            f"- Priority Level: {query.priority_level.title()}"
        ]
        
        if query.focus_areas:
            prompt_parts.append(f"- Focus Areas: {', '.join(query.focus_areas)}")
        
        if query.time_period:
            prompt_parts.append(f"- Time Period: {query.time_period}")
        
        if query.context:
            prompt_parts.extend([
                "",
                "Additional Context:",
                query.context
            ])
        
        prompt_parts.extend([
            "",
            "Please provide a comprehensive legal research analysis that includes:",
            "1. Key legal principles and concepts relevant to this question",
            "2. Applicable statutes, regulations, or constitutional provisions",
            "3. Relevant case law and precedents (cite specific cases when possible)",
            "4. Legal analysis and reasoning",
            "5. Potential counterarguments or alternative interpretations",
            "6. Practical implications and recommendations",
            "",
            "Format your response with clear headings and cite all sources mentioned.",
            "Be thorough but concise, focusing on the most relevant and authoritative sources.",
            "<|im_end|>",
            "<|im_start|>assistant"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_research_response(self, response: str, query_id: str) -> ResearchResult:
        """Parse LLM response into structured research result"""
        
        # Extract cited sources (basic pattern matching)
        sources = []
        lines = response.split('\n')
        
        for line in lines:
            # Look for common citation patterns
            if any(indicator in line.lower() for indicator in [
                'v.', 'vs.', 'case:', 'statute:', 'u.s.c.', 'f.2d', 'f.3d', 
                'supreme court', 'court of appeals', 'district court'
            ]):
                sources.append(line.strip())
        
        # Calculate confidence based on response quality indicators
        confidence = self._calculate_confidence(response)
        
        # Extract warnings and suggestions
        warnings = self._extract_warnings(response)
        suggestions = self._extract_suggestions(response)
        
        return ResearchResult(
            query_id=query_id,
            findings=response,
            sources_cited=sources,
            confidence_score=confidence,
            research_depth=1,
            warnings=warnings,
            follow_up_suggestions=suggestions
        )
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response quality indicators"""
        
        confidence_factors = {
            "has_citations": 0.3,
            "structured_format": 0.2,
            "legal_terminology": 0.2,
            "comprehensive_length": 0.15,
            "specific_references": 0.15
        }
        
        score = 0.0
        
        # Check for citations
        if any(indicator in response.lower() for indicator in [
            'v.', 'case:', 'statute:', 'court', 'cite'
        ]):
            score += confidence_factors["has_citations"]
        
        # Check for structured format
        if response.count('\n\n') > 2 and any(header in response for header in [
            '1.', '2.', '##', 'Analysis:', 'Conclusion:'
        ]):
            score += confidence_factors["structured_format"]
        
        # Check for legal terminology
        legal_terms = [
            'jurisdiction', 'precedent', 'statute', 'constitutional', 
            'ruling', 'court', 'legal', 'law', 'case'
        ]
        if sum(1 for term in legal_terms if term in response.lower()) >= 3:
            score += confidence_factors["legal_terminology"]
        
        # Check comprehensiveness (length as proxy)
        if len(response) > 1000:
            score += confidence_factors["comprehensive_length"]
        
        # Check for specific case/statute references
        if any(pattern in response for pattern in [
            'U.S.', 'F.2d', 'F.3d', 'S.Ct.', 'U.S.C.'
        ]):
            score += confidence_factors["specific_references"]
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_warnings(self, response: str) -> List[str]:
        """Extract potential warnings from research response"""
        warnings = []
        
        response_lower = response.lower()
        
        # Check for uncertainty indicators
        uncertainty_phrases = [
            "however", "but", "although", "unclear", "uncertain", 
            "may vary", "depends on", "could be", "might be"
        ]
        
        if sum(1 for phrase in uncertainty_phrases if phrase in response_lower) > 3:
            warnings.append("Research contains multiple uncertainty indicators")
        
        # Check for complexity warnings
        if "complex" in response_lower or "complicated" in response_lower:
            warnings.append("Legal issue appears complex - consider additional research")
        
        # Check for jurisdiction-specific warnings
        if "varies by jurisdiction" in response_lower:
            warnings.append("Results may vary by jurisdiction")
        
        return warnings
    
    def _extract_suggestions(self, response: str) -> List[str]:
        """Extract follow-up research suggestions"""
        suggestions = []
        
        # Look for explicit recommendations
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in [
                "recommend", "suggest", "consider", "should also", "further research"
            ]):
                suggestions.append(line.strip())
        
        # Add standard follow-up suggestions based on content
        if "constitutional" in response.lower():
            suggestions.append("Consider reviewing recent constitutional law developments")
        
        if "precedent" in response.lower():
            suggestions.append("Verify precedent status and any subsequent rulings")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def research(self, query: Union[str, ResearchQuery]) -> ResearchResult:
        """
        Conduct legal research based on query
        
        Args:
            query: Research question (string) or structured ResearchQuery
            
        Returns:
            ResearchResult: Comprehensive research findings
        """
        
        # Convert string to ResearchQuery if needed
        if isinstance(query, str):
            query = ResearchQuery(question=query)
        
        # Generate unique query ID
        query_id = self._generate_query_id(query)
        
        logger.info(f"Starting research for query: {query_id}")
        logger.debug(f"Query details: {query.to_dict()}")
        
        try:
            # Build research prompt
            prompt = self._build_research_prompt(query)
            
            # Generate research using LLM
            start_time = time.time()
            
            response = self.llm.generate(
                prompt,
                **self.llm_config
            )
            
            research_time = time.time() - start_time
            logger.info(f"Research completed in {research_time:.2f} seconds")
            
            # Parse and structure the response
            result = self._parse_research_response(response, query_id)
            
            logger.info(f"Research result - Confidence: {result.confidence_score:.2f}, "
                       f"Sources: {len(result.sources_cited)}, "
                       f"Warnings: {len(result.warnings)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Research failed for query {query_id}: {str(e)}")
            
            # Return error result
            return ResearchResult(
                query_id=query_id,
                findings=f"Research failed due to error: {str(e)}",
                confidence_score=0.0,
                warnings=["Research generation failed"]
            )
    
    def multi_angle_research(self, 
                           base_query: Union[str, ResearchQuery], 
                           perspectives: List[str] = None) -> List[ResearchResult]:
        """
        Conduct research from multiple angles/perspectives
        
        Args:
            base_query: Base research question
            perspectives: List of research perspectives to explore
            
        Returns:
            List[ResearchResult]: Results from each perspective
        """
        
        if perspectives is None:
            perspectives = [
                "plaintiff's perspective",
                "defendant's perspective", 
                "precedent analysis",
                "policy implications"
            ]
        
        results = []
        
        for perspective in perspectives:
            # Create modified query for this perspective
            if isinstance(base_query, str):
                modified_query = ResearchQuery(
                    question=f"{base_query} (from {perspective})",
                    context=f"Focus on {perspective}"
                )
            else:
                modified_query = ResearchQuery(
                    question=f"{base_query.question} (from {perspective})",
                    research_type=base_query.research_type,
                    jurisdiction=base_query.jurisdiction,
                    focus_areas=base_query.focus_areas,
                    context=f"{base_query.context or ''} Focus on {perspective}".strip()
                )
            
            result = self.research(modified_query)
            results.append(result)
        
        logger.info(f"Completed multi-angle research with {len(results)} perspectives")
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and configuration"""
        return {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "llm_loaded": self.llm.is_loaded,
            "max_research_depth": self.max_research_depth,
            "require_citations": self.require_citations,
            "llm_config": self.llm_config,
            "model_info": self.llm.get_model_info()
        }

# Convenience function for easy importing
def create_researcher() -> LegalResearcher:
    """Create and return a new LegalResearcher instance"""
    return LegalResearcher()