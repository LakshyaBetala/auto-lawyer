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

# Configure logging
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Types of legal documents that can be drafted"""
    MOTION = "motion"
    BRIEF = "brief"
    MEMORANDUM = "memorandum"
    PLEADING = "pleading"
    CONTRACT = "contract"
    OPINION_LETTER = "opinion_letter"
    SETTLEMENT_AGREEMENT = "settlement_agreement"
    DISCOVERY_REQUEST = "discovery_request"
    RESPONSE = "response"
    APPEAL = "appeal"
    GENERAL = "general"

class ArgumentStyle(Enum):
    """Argument presentation styles"""
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    ACADEMIC = "academic"
    PERSUASIVE = "persuasive"
    FACTUAL = "factual"

class CitationStyle(Enum):
    """Legal citation formats"""
    BLUEBOOK = "bluebook"
    ALWD = "alwd"
    MLA = "mla"
    APA = "apa"
    CHICAGO = "chicago"

@dataclass
class DraftingRequest:
    """Structured request for document drafting"""
    document_type: DocumentType
    title: str
    objective: str
    research_results: List[ResearchResult] = field(default_factory=list)
    
    # Formatting preferences
    citation_style: CitationStyle = CitationStyle.BLUEBOOK
    argument_style: ArgumentStyle = ArgumentStyle.MODERATE
    max_length_words: Optional[int] = None
    
    # Content specifications
    key_arguments: List[str] = field(default_factory=list)
    counterarguments: List[str] = field(default_factory=list)
    factual_background: Optional[str] = None
    procedural_history: Optional[str] = None
    
    # Client/case information
    client_name: Optional[str] = None
    opposing_party: Optional[str] = None
    case_number: Optional[str] = None
    court: Optional[str] = None
    jurisdiction: Optional[str] = None
    
    # Additional context
    urgency: str = "normal"  # low, normal, high, emergency
    special_instructions: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "document_type": self.document_type.value,
            "title": self.title,
            "objective": self.objective,
            "citation_style": self.citation_style.value,
            "argument_style": self.argument_style.value,
            "max_length_words": self.max_length_words,
            "key_arguments": self.key_arguments,
            "counterarguments": self.counterarguments,
            "urgency": self.urgency,
            "research_results_count": len(self.research_results)
        }

@dataclass
class DraftResult:
    """Result of document drafting with metadata"""
    draft_id: str
    document_type: DocumentType
    title: str
    content: str
    
    # Quality metrics
    word_count: int = 0
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    drafting_time_seconds: float = 0.0
    revision_number: int = 1
    
    # Analysis
    key_points_covered: List[str] = field(default_factory=list)
    citations_included: List[str] = field(default_factory=list)
    suggestions_for_improvement: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "draft_id": self.draft_id,
            "document_type": self.document_type.value,
            "title": self.title,
            "content": self.content,
            "word_count": self.word_count,
            "confidence_score": self.confidence_score,
            "completeness_score": self.completeness_score,
            "timestamp": self.timestamp.isoformat(),
            "drafting_time_seconds": self.drafting_time_seconds,
            "revision_number": self.revision_number,
            "key_points_covered": self.key_points_covered,
            "citations_included": self.citations_included,
            "suggestions_for_improvement": self.suggestions_for_improvement,
            "potential_issues": self.potential_issues
        }

class LegalDrafter:
    """
    Legal Document Drafter Agent powered by local LLM
    Creates professional legal documents based on research and specifications
    """
    
    def __init__(self):
        """Initialize the Legal Drafter agent"""
        self.config = get_config()
        self.llm = get_llm_runner()
        self.agent_name = "Legal Drafter"
        self.agent_id = "drafter"
        
        # Drafting configuration
        self.default_citation_style = CitationStyle(self.config.legal.default_citation_style)
        
        # Load agent-specific LLM settings
        self.llm_config = self._load_agent_config()
        
        # Document templates
        self.templates = self._load_document_templates()
        
        logger.info(f"{self.agent_name} initialized successfully")
    
    def _load_agent_config(self) -> Dict[str, Any]:
        """Load drafter-specific LLM configuration"""
        try:
            with open(self.config.llm.model_config_path, 'r') as f:
                model_config = json.load(f)
            
            return model_config.get("agent_overrides", {}).get("drafter", {})
        except Exception as e:
            logger.warning(f"Could not load agent config: {e}. Using defaults.")
            return {
                "temperature": 0.3,
                "max_tokens": 4096,
                "top_p": 0.9
            }
    
    def _load_document_templates(self) -> Dict[str, str]:
        """Load document structure templates"""
        # Basic templates for different document types
        return {
            "motion": """
[COURT HEADER]

{title}

TO THE HONORABLE COURT:

I. INTRODUCTION
{introduction}

II. FACTUAL BACKGROUND
{factual_background}

III. ARGUMENT
{main_arguments}

IV. CONCLUSION
{conclusion}

Respectfully submitted,
[ATTORNEY SIGNATURE BLOCK]
            """,
            
            "brief": """
[COURT HEADER]

{title}

TABLE OF CONTENTS
[TABLE OF CONTENTS]

TABLE OF AUTHORITIES
[TABLE OF AUTHORITIES]

I. STATEMENT OF THE CASE
{case_statement}

II. FACTUAL BACKGROUND
{factual_background}

III. ARGUMENT
{main_arguments}

IV. CONCLUSION
{conclusion}

Respectfully submitted,
[ATTORNEY SIGNATURE BLOCK]
            """,
            
            "memorandum": """
MEMORANDUM

TO: {recipient}
FROM: {author}
DATE: {date}
RE: {subject}

EXECUTIVE SUMMARY
{executive_summary}

FACTUAL BACKGROUND
{factual_background}

LEGAL ANALYSIS
{legal_analysis}

RECOMMENDATION
{recommendation}
            """,
            
            "general": """
{title}

{content}
            """
        }
    
    def _generate_draft_id(self, request: DraftingRequest) -> str:
        """Generate unique ID for draft"""
        timestamp = int(time.time())
        doc_hash = hash(f"{request.title}_{request.objective}")
        return f"draft_{request.document_type.value}_{timestamp}_{abs(doc_hash)}"
    
    def _build_drafting_prompt(self, request: DraftingRequest) -> str:
        """Build comprehensive drafting prompt for the LLM"""
        
        # Get system prompt from config
        try:
            with open(self.config.llm.model_config_path, 'r') as f:
                model_config = json.load(f)
            system_prompt = model_config.get("system_prompts", {}).get("drafter", "")
        except:
            system_prompt = "<|im_start|>system\nYou are a professional legal document drafter.<|im_end|>"
        
        prompt_parts = [
            system_prompt,
            "<|im_start|>user",
            f"Draft a {request.document_type.value.replace('_', ' ')} with the following specifications:",
            "",
            f"Document Title: {request.title}",
            f"Objective: {request.objective}",
            f"Citation Style: {request.citation_style.value.title()}",
            f"Argument Style: {request.argument_style.value.title()}",
        ]
        
        # Add length constraint if specified
        if request.max_length_words:
            prompt_parts.append(f"Maximum Length: {request.max_length_words} words")
        
        # Add case information if provided
        if any([request.client_name, request.opposing_party, request.case_number]):
            prompt_parts.append("\nCase Information:")
            if request.client_name:
                prompt_parts.append(f"- Client: {request.client_name}")
            if request.opposing_party:
                prompt_parts.append(f"- Opposing Party: {request.opposing_party}")
            if request.case_number:
                prompt_parts.append(f"- Case Number: {request.case_number}")
            if request.court:
                prompt_parts.append(f"- Court: {request.court}")
            if request.jurisdiction:
                prompt_parts.append(f"- Jurisdiction: {request.jurisdiction}")
        
        # Add factual background if provided
        if request.factual_background:
            prompt_parts.extend([
                "\nFactual Background:",
                request.factual_background
            ])
        
        # Add procedural history if provided
        if request.procedural_history:
            prompt_parts.extend([
                "\nProcedural History:",
                request.procedural_history
            ])
        
        # Add key arguments
        if request.key_arguments:
            prompt_parts.extend([
                "\nKey Arguments to Include:",
                *[f"- {arg}" for arg in request.key_arguments]
            ])
        
        # Add counterarguments to address
        if request.counterarguments:
            prompt_parts.extend([
                "\nCounterarguments to Address:",
                *[f"- {counter}" for counter in request.counterarguments]
            ])
        
        # Add research findings
        if request.research_results:
            prompt_parts.append("\nResearch Findings to Incorporate:")
            for i, result in enumerate(request.research_results, 1):
                prompt_parts.extend([
                    f"\nResearch Finding #{i}:",
                    f"Query: {result.query_id}",
                    f"Findings: {result.findings[:500]}..." if len(result.findings) > 500 else f"Findings: {result.findings}",
                    f"Sources: {', '.join(result.sources_cited[:3])}" if result.sources_cited else "Sources: None specified"
                ])
        
        # Add special instructions
        if request.special_instructions:
            prompt_parts.extend([
                "\nSpecial Instructions:",
                request.special_instructions
            ])
        
        prompt_parts.extend([
            "\nDrafting Requirements:",
            "1. Use proper legal document structure and formatting",
            "2. Include appropriate legal citations where applicable",
            "3. Ensure logical flow and persuasive argumentation",
            "4. Use professional legal language and tone",
            "5. Address potential counterarguments",
            "6. Include relevant case law and statutory authority",
            "",
            "Please draft a complete, professional legal document that meets these specifications.",
            "<|im_end|>",
            "<|im_start|>assistant"
        ])
        
        return "\n".join(prompt_parts)
    
    def _analyze_draft_quality(self, content: str, request: DraftingRequest) -> Dict[str, Any]:
        """Analyze draft quality and completeness"""
        
        analysis = {
            "word_count": len(content.split()),
            "confidence_score": 0.0,
            "completeness_score": 0.0,
            "key_points_covered": [],
            "citations_included": [],
            "suggestions": [],
            "issues": []
        }
        
        # Calculate word count
        words = content.split()
        analysis["word_count"] = len(words)
        
        # Check for legal document structure
        structure_indicators = [
            "introduction", "background", "argument", "conclusion",
            "statement", "analysis", "recommendation"
        ]
        
        structure_score = sum(1 for indicator in structure_indicators 
                            if indicator.lower() in content.lower()) / len(structure_indicators)
        
        # Check for legal language quality
        legal_terms = [
            "court", "plaintiff", "defendant", "jurisdiction", "statute",
            "precedent", "holding", "ruling", "motion", "brief"
        ]
        
        legal_language_score = min(
            sum(1 for term in legal_terms if term.lower() in content.lower()) / 10, 1.0
        )
        
        # Extract citations
        citation_patterns = [
            r'\d+\s+U\.S\.\s+\d+',  # Supreme Court
            r'\d+\s+F\.\d+d\s+\d+',  # Federal courts
            r'\d+\s+U\.S\.C\.\s+§\s*\d+',  # USC
            r'v\.\s+\w+',  # Case names
        ]
        
        for pattern in citation_patterns:
            citations = re.findall(pattern, content)
            analysis["citations_included"].extend(citations)
        
        # Check coverage of key arguments
        for arg in request.key_arguments:
            if any(word.lower() in content.lower() for word in arg.split()):
                analysis["key_points_covered"].append(arg)
        
        # Calculate confidence score
        factors = {
            "structure": structure_score * 0.3,
            "legal_language": legal_language_score * 0.3,
            "citations": min(len(analysis["citations_included"]) / 5, 1.0) * 0.2,
            "length": min(analysis["word_count"] / 500, 1.0) * 0.2
        }
        
        analysis["confidence_score"] = sum(factors.values())
        
        # Calculate completeness score
        completeness_factors = {
            "key_points": len(analysis["key_points_covered"]) / max(len(request.key_arguments), 1),
            "structure": structure_score,
            "minimum_length": 1.0 if analysis["word_count"] >= 250 else analysis["word_count"] / 250
        }
        
        analysis["completeness_score"] = sum(completeness_factors.values()) / len(completeness_factors)
        
        # Generate suggestions
        if analysis["confidence_score"] < 0.7:
            analysis["suggestions"].append("Consider adding more legal citations and authority")
        
        if analysis["completeness_score"] < 0.8:
            analysis["suggestions"].append("Review coverage of all key arguments")
        
        if analysis["word_count"] < 200:
            analysis["suggestions"].append("Document may benefit from additional detail and analysis")
        
        # Check for potential issues
        if not analysis["citations_included"]:
            analysis["issues"].append("No legal citations detected - consider adding supporting authority")
        
        if request.max_length_words and analysis["word_count"] > request.max_length_words:
            analysis["issues"].append(f"Document exceeds requested length limit of {request.max_length_words} words")
        
        return analysis
    
    def draft_document(self, request: DraftingRequest) -> DraftResult:
        """
        Draft a legal document based on specifications
        
        Args:
            request: DraftingRequest with specifications
            
        Returns:
            DraftResult: Complete draft with analysis
        """
        
        draft_id = self._generate_draft_id(request)
        
        logger.info(f"Starting document drafting for: {request.title}")
        logger.debug(f"Draft request: {request.to_dict()}")
        
        try:
            # Build drafting prompt
            prompt = self._build_drafting_prompt(request)
            
            # Generate draft using LLM
            start_time = time.time()
            
            content = self.llm.generate(
                prompt,
                **self.llm_config
            )
            
            drafting_time = time.time() - start_time
            
            # Analyze draft quality
            analysis = self._analyze_draft_quality(content, request)
            
            # Create result
            result = DraftResult(
                draft_id=draft_id,
                document_type=request.document_type,
                title=request.title,
                content=content,
                word_count=analysis["word_count"],
                confidence_score=analysis["confidence_score"],
                completeness_score=analysis["completeness_score"],
                drafting_time_seconds=drafting_time,
                key_points_covered=analysis["key_points_covered"],
                citations_included=analysis["citations_included"],
                suggestions_for_improvement=analysis["suggestions"],
                potential_issues=analysis["issues"]
            )
            
            logger.info(f"Draft completed - ID: {draft_id}, "
                       f"Words: {result.word_count}, "
                       f"Confidence: {result.confidence_score:.2f}, "
                       f"Time: {drafting_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Drafting failed for {draft_id}: {str(e)}")
            
            return DraftResult(
                draft_id=draft_id,
                document_type=request.document_type,
                title=request.title,
                content=f"Draft generation failed: {str(e)}",
                confidence_score=0.0,
                potential_issues=["Draft generation failed"]
            )
    
    def revise_document(self, 
                       original_draft: DraftResult, 
                       revision_instructions: str,
                       research_updates: List[ResearchResult] = None) -> DraftResult:
        """
        Revise an existing document based on feedback
        
        Args:
            original_draft: Original DraftResult to revise
            revision_instructions: Specific revision instructions
            research_updates: Additional research to incorporate
            
        Returns:
            DraftResult: Revised document
        """
        
        logger.info(f"Revising document: {original_draft.draft_id}")
        
        try:
            # Build revision prompt
            revision_prompt_parts = [
                "<|im_start|>system",
                "You are a legal document drafter revising an existing document based on specific feedback.",
                "<|im_end|>",
                "<|im_start|>user",
                "Please revise the following legal document based on the provided instructions:",
                "",
                "Original Document:",
                "=" * 50,
                original_draft.content,
                "=" * 50,
                "",
                "Revision Instructions:",
                revision_instructions
            ]
            
            # Add new research if provided
            if research_updates:
                revision_prompt_parts.extend([
                    "",
                    "Additional Research to Incorporate:"
                ])
                for result in research_updates:
                    revision_prompt_parts.extend([
                        f"- {result.findings[:300]}..."
                    ])
            
            revision_prompt_parts.extend([
                "",
                "Please provide the complete revised document that addresses the revision instructions.",
                "<|im_end|>",
                "<|im_start|>assistant"
            ])
            
            revision_prompt = "\n".join(revision_prompt_parts)
            
            # Generate revision
            start_time = time.time()
            revised_content = self.llm.generate(revision_prompt, **self.llm_config)
            revision_time = time.time() - start_time
            
            # Create revised result
            revised_result = DraftResult(
                draft_id=f"{original_draft.draft_id}_rev_{original_draft.revision_number + 1}",
                document_type=original_draft.document_type,
                title=original_draft.title,
                content=revised_content,
                drafting_time_seconds=revision_time,
                revision_number=original_draft.revision_number + 1
            )
            
            # Analyze revised draft (simplified analysis for revision)
            revised_result.word_count = len(revised_content.split())
            
            logger.info(f"Revision completed - New ID: {revised_result.draft_id}")
            
            return revised_result
            
        except Exception as e:
            logger.error(f"Revision failed: {str(e)}")
            return original_draft  # Return original if revision fails
    
    def get_document_templates(self) -> Dict[str, str]:
        """Get available document templates"""
        return self.templates.copy()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and configuration"""
        return {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "llm_loaded": self.llm.is_loaded,
            "default_citation_style": self.default_citation_style.value,
            "available_document_types": [dt.value for dt in DocumentType],
            "available_argument_styles": [ast.value for ast in ArgumentStyle],
            "llm_config": self.llm_config,
            "templates_loaded": len(self.templates)
        }

# Convenience function for easy importing
def create_drafter() -> LegalDrafter:
    """Create and return a new LegalDrafter instance"""
    return LegalDrafter()