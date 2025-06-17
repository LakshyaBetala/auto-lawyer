import logging
import json
import time
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from backend.local_llm.llm_runner import get_llm_runner
from backend.config import get_config
from backend.agents.researcher import ResearchResult
from backend.agents.drafter import DraftResult

# Configure logging
logger = logging.getLogger(__name__)

class SummaryType(Enum):
    """Types of summaries that can be generated"""
    EXECUTIVE = "executive"           # High-level executive summary
    RESEARCH = "research"            # Research findings summary
    DOCUMENT = "document"            # Legal document summary
    CASE = "case"                   # Case analysis summary
    ISSUE = "issue"                 # Legal issue summary
    PROCEDURAL = "procedural"       # Procedural history summary
    FACTUAL = "factual"            # Factual background summary
    ARGUMENTATIVE = "argumentative" # Arguments and counterarguments
    COMPREHENSIVE = "comprehensive" # Full comprehensive summary

class SummaryLength(Enum):
    """Summary length options"""
    BRIEF = "brief"           # 1-2 paragraphs (100-200 words)
    SHORT = "short"           # 2-3 paragraphs (200-400 words)
    MEDIUM = "medium"         # 3-5 paragraphs (400-700 words)
    LONG = "long"            # 5-8 paragraphs (700-1200 words)
    DETAILED = "detailed"     # 8+ paragraphs (1200+ words)

class AudienceLevel(Enum):
    """Target audience complexity levels"""
    GENERAL_PUBLIC = "general_public"     # Layperson-friendly
    CLIENT = "client"                     # Educated client
    ATTORNEY = "attorney"                 # Legal professional
    JUDGE = "judge"                      # Judicial audience
    ACADEMIC = "academic"                # Academic/scholarly
    EXPERT = "expert"                    # Subject matter expert

@dataclass
class SummaryRequest:
    """Structured request for document/research summarization"""
    summary_type: SummaryType
    length: SummaryLength = SummaryLength.MEDIUM
    audience: AudienceLevel = AudienceLevel.ATTORNEY
    
    # Content to summarize
    research_results: List[ResearchResult] = field(default_factory=list)
    draft_results: List[DraftResult] = field(default_factory=list)
    raw_text: Optional[str] = None
    
    # Summary focus
    key_focus_areas: List[str] = field(default_factory=list)
    include_citations: bool = True
    include_recommendations: bool = True
    include_next_steps: bool = True
    
    # Formatting preferences
    use_bullet_points: bool = False
    include_headings: bool = True
    highlight_risks: bool = True
    highlight_opportunities: bool = True
    
    # Context
    case_context: Optional[str] = None
    urgency: str = "normal"  # low, normal, high, critical
    special_instructions: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "summary_type": self.summary_type.value,
            "length": self.length.value,
            "audience": self.audience.value,
            "research_results_count": len(self.research_results),
            "draft_results_count": len(self.draft_results),
            "has_raw_text": bool(self.raw_text),
            "key_focus_areas": self.key_focus_areas,
            "include_citations": self.include_citations,
            "include_recommendations": self.include_recommendations
        }

@dataclass
class SummaryResult:
    """Result of summarization with metadata"""
    summary_id: str
    summary_type: SummaryType
    title: str
    content: str
    
    # Metrics
    word_count: int = 0
    compression_ratio: float = 0.0  # original_words / summary_words
    readability_score: float = 0.0
    
    # Content analysis
    key_points_extracted: List[str] = field(default_factory=list)
    citations_preserved: List[str] = field(default_factory=list)
    recommendations_included: List[str] = field(default_factory=list)
    risks_identified: List[str] = field(default_factory=list)
    opportunities_noted: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_seconds: float = 0.0
    confidence_score: float = 0.0
    
    # Quality indicators
    completeness_score: float = 0.0
    accuracy_indicators: List[str] = field(default_factory=list)
    potential_gaps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "summary_id": self.summary_id,
            "summary_type": self.summary_type.value,
            "title": self.title,
            "content": self.content,
            "word_count": self.word_count,
            "compression_ratio": self.compression_ratio,
            "readability_score": self.readability_score,
            "key_points_extracted": self.key_points_extracted,
            "citations_preserved": self.citations_preserved,
            "recommendations_included": self.recommendations_included,
            "risks_identified": self.risks_identified,
            "opportunities_noted": self.opportunities_noted,
            "timestamp": self.timestamp.isoformat(),
            "processing_time_seconds": self.processing_time_seconds,
            "confidence_score": self.confidence_score,
            "completeness_score": self.completeness_score
        }

class LegalSummarizer:
    """
    Legal Document Summarizer Agent powered by local LLM
    Creates concise, accurate summaries of research and legal documents
    """
    
    def __init__(self):
        """Initialize the Legal Summarizer agent"""
        self.config = get_config()
        self.llm = get_llm_runner()
        self.agent_name = "Legal Summarizer"
        self.agent_id = "summarizer"
        
        # Load agent-specific LLM settings
        self.llm_config = self._load_agent_config()
        
        # Summary length guidelines (target word counts)
        self.length_guidelines = {
            SummaryLength.BRIEF: (100, 200),
            SummaryLength.SHORT: (200, 400),
            SummaryLength.MEDIUM: (400, 700),
            SummaryLength.LONG: (700, 1200),
            SummaryLength.DETAILED: (1200, 2000)
        }
        
        logger.info(f"{self.agent_name} initialized successfully")
    
    def _load_agent_config(self) -> Dict[str, Any]:
        """Load summarizer-specific LLM configuration"""
        try:
            with open(self.config.llm.model_config_path, 'r') as f:
                model_config = json.load(f)
            
            return model_config.get("agent_overrides", {}).get("summarizer", {})
        except Exception as e:
            logger.warning(f"Could not load agent config: {e}. Using defaults.")
            return {
                "temperature": 0.05,
                "max_tokens": 1024,
                "top_p": 0.95
            }
    
    def _generate_summary_id(self, request: SummaryRequest) -> str:
        """Generate unique ID for summary"""
        timestamp = int(time.time())
        content_hash = hash(f"{request.summary_type.value}_{len(request.research_results)}_{len(request.draft_results)}")
        return f"summary_{request.summary_type.value}_{timestamp}_{abs(content_hash)}"
    
    def _prepare_content_for_summarization(self, request: SummaryRequest) -> str:
        """Prepare and combine all content for summarization"""
        content_parts = []
        
        # Add research results
        if request.research_results:
            content_parts.append("=== RESEARCH FINDINGS ===")
            for i, result in enumerate(request.research_results, 1):
                content_parts.extend([
                    f"\nResearch Query #{i}: {result.query_id}",
                    f"Findings: {result.findings}",
                    f"Sources: {', '.join(result.sources_cited[:5])}" if result.sources_cited else "Sources: None specified",
                    f"Confidence: {result.confidence_score:.2f}"
                ])
        
        # Add draft results
        if request.draft_results:
            content_parts.append("\n=== DRAFTED DOCUMENTS ===")
            for i, draft in enumerate(request.draft_results, 1):
                content_parts.extend([
                    f"\nDraft #{i}: {draft.title}",
                    f"Document Type: {draft.document_type.value.replace('_', ' ').title()}",
                    f"Content: {draft.content}",
                    f"Word Count: {draft.word_count}",
                    f"Confidence: {draft.confidence_score:.2f}"
                ])
        
        # Add raw text
        if request.raw_text:
            content_parts.extend([
                "\n=== ADDITIONAL CONTENT ===",
                request.raw_text
            ])
        
        return "\n".join(content_parts)
    
    def _build_summarization_prompt(self, request: SummaryRequest, content: str) -> str:
        """Build comprehensive summarization prompt for the LLM"""
        
        # Get system prompt from config
        try:
            with open(self.config.llm.model_config_path, 'r') as f:
                model_config = json.load(f)
            system_prompt = model_config.get("system_prompts", {}).get("summarizer", "")
        except:
            system_prompt = "<|im_start|>system\nYou are a legal document summarizer.<|im_end|>"
        
        # Get target word count
        min_words, max_words = self.length_guidelines[request.length]
        
        prompt_parts = [
            system_prompt,
            "<|im_start|>user",
            f"Create a {request.summary_type.value.replace('_', ' ')} summary of the following legal content:",
            "",
            "Summary Specifications:",
            f"- Type: {request.summary_type.value.replace('_', ' ').title()}",
            f"- Length: {request.length.value.title()} ({min_words}-{max_words} words)",
            f"- Audience: {request.audience.value.replace('_', ' ').title()}",
            f"- Include Citations: {'Yes' if request.include_citations else 'No'}",
            f"- Include Recommendations: {'Yes' if request.include_recommendations else 'No'}",
            f"- Include Next Steps: {'Yes' if request.include_next_steps else 'No'}",
            f"- Use Bullet Points: {'Yes' if request.use_bullet_points else 'No'}",
            f"- Include Headings: {'Yes' if request.include_headings else 'No'}"
        ]
        
        # Add focus areas
        if request.key_focus_areas:
            prompt_parts.extend([
                "",
                "Key Focus Areas:",
                *[f"- {area}" for area in request.key_focus_areas]
            ])
        
        # Add case context
        if request.case_context:
            prompt_parts.extend([
                "",
                "Case Context:",
                request.case_context
            ])
        
        # Add special instructions
        if request.special_instructions:
            prompt_parts.extend([
                "",
                "Special Instructions:",
                request.special_instructions
            ])
        
        # Add content to summarize
        prompt_parts.extend([
            "",
            "CONTENT TO SUMMARIZE:",
            "=" * 60,
            content,
            "=" * 60,
            ""
        ])
        
        # Add specific summarization requirements based on audience
        audience_instructions = {
            AudienceLevel.GENERAL_PUBLIC: "Use clear, simple language. Avoid legal jargon. Explain legal concepts in plain English.",
            AudienceLevel.CLIENT: "Use professional but accessible language. Briefly explain legal terms when necessary.",
            AudienceLevel.ATTORNEY: "Use appropriate legal terminology. Focus on legal implications and strategic considerations.",
            AudienceLevel.JUDGE: "Use formal legal language. Focus on legal reasoning and applicable law.",
            AudienceLevel.ACADEMIC: "Use scholarly tone. Include theoretical and analytical perspectives.",
            AudienceLevel.EXPERT: "Use technical language appropriate for subject matter experts."
        }
        
        prompt_parts.extend([
            f"Audience-Specific Instructions: {audience_instructions[request.audience]}",
            ""
        ])
        
        # Add summarization requirements
        requirements = [
            f"1. Create a well-structured summary within {min_words}-{max_words} words",
            "2. Preserve the most important legal points and arguments",
            "3. Maintain accuracy and avoid misrepresentation"
        ]
        
        if request.include_citations:
            requirements.append("4. Include key citations and legal authorities")
        
        if request.include_recommendations:
            requirements.append("5. Provide actionable recommendations where appropriate")
        
        if request.highlight_risks:
            requirements.append("6. Highlight potential legal risks and concerns")
        
        if request.highlight_opportunities:
            requirements.append("7. Identify opportunities and favorable aspects")
        
        prompt_parts.extend([
            "Summarization Requirements:",
            *requirements,
            "",
            "Please provide a comprehensive, accurate summary that meets these specifications.",
            "<|im_end|>",
            "<|im_start|>assistant"
        ])
        
        return "\n".join(prompt_parts)
    
    def _analyze_summary_quality(self, 
                                summary: str, 
                                original_content: str, 
                                request: SummaryRequest) -> Dict[str, Any]:
        """Analyze summary quality and extract metadata"""
        
        analysis = {
            "word_count": len(summary.split()),
            "compression_ratio": 0.0,
            "readability_score": 0.0,
            "key_points": [],
            "citations": [],
            "recommendations": [],
            "risks": [],
            "opportunities": [],
            "confidence_score": 0.0,
            "completeness_score": 0.0,
            "accuracy_indicators": [],
            "potential_gaps": []
        }
        
        # Calculate compression ratio
        original_words = len(original_content.split())
        summary_words = len(summary.split())
        if summary_words > 0:
            analysis["compression_ratio"] = original_words / summary_words
        
        analysis["word_count"] = summary_words
        
        # Extract citations
        citation_patterns = [
            r'\d+\s+U\.S\.\s+\d+',
            r'\d+\s+F\.\d+d\s+\d+',
            r'\d+\s+U\.S\.C\.\s+§\s*\d+',
            r'v\.\s+\w+',
        ]
        
        for pattern in citation_patterns:
            citations = re.findall(pattern, summary)
            analysis["citations"].extend(citations)
        
        # Extract key points (sentences with legal indicators)
        sentences = re.split(r'[.!?]+', summary)
        legal_indicators = ['court', 'ruling', 'statute', 'case', 'law', 'legal', 'jurisdiction']
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in legal_indicators):
                if len(sentence.strip()) > 20:  # Meaningful length
                    analysis["key_points"].append(sentence.strip())
        
        # Extract recommendations (look for recommendation language)
        recommendation_indicators = ['recommend', 'should', 'suggest', 'advise', 'propose']
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in recommendation_indicators):
                analysis["recommendations"].append(sentence.strip())
        
        # Extract risks (look for risk language)
        risk_indicators = ['risk', 'danger', 'concern', 'problem', 'issue', 'challenge', 'potential']
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in risk_indicators):
                analysis["risks"].append(sentence.strip())
        
        # Extract opportunities (look for positive language)
        opportunity_indicators = ['opportunity', 'advantage', 'benefit', 'favorable', 'strong', 'likely to succeed']
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in opportunity_indicators):
                analysis["opportunities"].append(sentence.strip())
        
        # Calculate confidence score
        target_min, target_max = self.length_guidelines[request.length]
        
        factors = {
            "length_appropriate": 1.0 if target_min <= summary_words <= target_max else 0.5,
            "citations_present": 1.0 if analysis["citations"] else 0.0 if request.include_citations else 1.0,
            "structure_quality": min(len(analysis["key_points"]) / 5, 1.0),
            "compression_reasonable": 1.0 if 3 <= analysis["compression_ratio"] <= 20 else 0.5
        }
        
        analysis["confidence_score"] = sum(factors.values()) / len(factors)
        
        # Calculate completeness score
        completeness_factors = {
            "key_points_coverage": min(len(analysis["key_points"]) / 3, 1.0),
            "recommendations": 1.0 if analysis["recommendations"] or not request.include_recommendations else 0.0,
            "appropriate_length": factors["length_appropriate"]
        }
        
        analysis["completeness_score"] = sum(completeness_factors.values()) / len(completeness_factors)
        
        # Quality indicators
        if analysis["citations"]:
            analysis["accuracy_indicators"].append("Contains legal citations")
        
        if analysis["compression_ratio"] > 3:
            analysis["accuracy_indicators"].append("Appropriate compression ratio")
        
        if len(analysis["key_points"]) >= 3:
            analysis["accuracy_indicators"].append("Multiple key points identified")
        
        # Potential gaps
        if request.include_citations and not analysis["citations"]:
            analysis["potential_gaps"].append("No legal citations included")
        
        if request.include_recommendations and not analysis["recommendations"]:
            analysis["potential_gaps"].append("No recommendations provided")
        
        if summary_words < target_min:
            analysis["potential_gaps"].append("Summary may be too brief")
        elif summary_words > target_max:
            analysis["potential_gaps"].append("Summary may be too lengthy")
        
        return analysis
    
    def summarize(self, request: SummaryRequest) -> SummaryResult:
        """
        Create a summary based on specifications
        
        Args:
            request: SummaryRequest with content and specifications
            
        Returns:
            SummaryResult: Complete summary with analysis
        """
        
        summary_id = self._generate_summary_id(request)
        
        logger.info(f"Starting summarization - Type: {request.summary_type.value}, "
                   f"Length: {request.length.value}")
        logger.debug(f"Summary request: {request.to_dict()}")
        
        try:
            # Prepare content for summarization
            content = self._prepare_content_for_summarization(request)
            
            if not content.strip():
                raise ValueError("No content provided for summarization")
            
            # Build summarization prompt
            prompt = self._build_summarization_prompt(request, content)
            
            # Generate summary using LLM
            start_time = time.time()
            
            summary_content = self.llm.generate(
                prompt,
                **self.llm_config
            )
            
            processing_time = time.time() - start_time
            
            # Analyze summary quality
            analysis = self._analyze_summary_quality(summary_content, content, request)
            
            # Generate title based on summary type
            title_map = {
                SummaryType.EXECUTIVE: "Executive Summary",
                SummaryType.RESEARCH: "Research Summary",
                SummaryType.DOCUMENT: "Document Summary",
                SummaryType.CASE: "Case Analysis Summary",
                SummaryType.ISSUE: "Legal Issue Summary",
                SummaryType.PROCEDURAL: "Procedural Summary",
                SummaryType.FACTUAL: "Factual Summary",
                SummaryType.ARGUMENTATIVE: "Arguments Summary",
                SummaryType.COMPREHENSIVE: "Comprehensive Summary"
            }
            
            # Create result
            result = SummaryResult(
                summary_id=summary_id,
                summary_type=request.summary_type,
                title=title_map.get(request.summary_type, "Summary"),
                content=summary_content,
                word_count=analysis["word_count"],
                compression_ratio=analysis["compression_ratio"],
                key_points_extracted=analysis["key_points"][:10],  # Limit to top 10
                citations_preserved=analysis["citations"],
                recommendations_included=analysis["recommendations"][:5],  # Limit to top 5
                risks_identified=analysis["risks"][:5],
                opportunities_noted=analysis["opportunities"][:5],
                processing_time_seconds=processing_time,
                confidence_score=analysis["confidence_score"],
                completeness_score=analysis["completeness_score"],
                accuracy_indicators=analysis["accuracy_indicators"],
                potential_gaps=analysis["potential_gaps"]
            )
            
            logger.info(f"Summary completed - ID: {summary_id}, "
                       f"Words: {result.word_count}, "
                       f"Compression: {result.compression_ratio:.1f}x, "
                       f"Confidence: {result.confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Summarization failed for {summary_id}: {str(e)}")
            
            return SummaryResult(
                summary_id=summary_id,
                summary_type=request.summary_type,
                title="Summary Generation Failed",
                content=f"Summary generation failed: {str(e)}",
                confidence_score=0.0,
                potential_gaps=["Summary generation failed"]
            )
    
    def create_multi_perspective_summary(self, 
                                       base_request: SummaryRequest,
                                       perspectives: List[str] = None) -> List[SummaryResult]:
        """
        Create summaries from multiple perspectives
        
        Args:
            base_request: Base summary request
            perspectives: List of perspectives to summarize from
            
        Returns:
            List[SummaryResult]: Summaries from each perspective
        """
        
        if perspectives is None:
            perspectives = [
                "legal strengths and advantages",
                "potential risks and weaknesses", 
                "strategic recommendations",
                "next steps and timeline"
            ]
        
        results = []
        
        for perspective in perspectives:
            # Create modified request for this perspective
            perspective_request = SummaryRequest(
                summary_type=base_request.summary_type,
                length=SummaryLength.SHORT,  # Shorter for multiple perspectives
                audience=base_request.audience,
                research_results=base_request.research_results,
                draft_results=base_request.draft_results,
                raw_text=base_request.raw_text,
                key_focus_areas=[perspective],
                include_citations=base_request.include_citations,
                special_instructions=f"Focus specifically on: {perspective}"
            )
            
            result = self.summarize(perspective_request)
            result.title = f"{result.title} - {perspective.title()}"
            results.append(result)
        
        logger.info(f"Completed multi-perspective summary with {len(results)} perspectives")
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and configuration"""
        return {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "llm_loaded": self.llm.is_loaded,
            "available_summary_types": [st.value for st in SummaryType],
            "available_lengths": [sl.value for sl in SummaryLength],
            "available_audiences": [al.value for al in AudienceLevel],
            "length_guidelines": {k.value: v for k, v in self.length_guidelines.items()},
            "llm_config": self.llm_config
        }

# Convenience function for easy importing
def create_summarizer() -> LegalSummarizer:
    """Create and return a new LegalSummarizer instance"""
    return LegalSummarizer()