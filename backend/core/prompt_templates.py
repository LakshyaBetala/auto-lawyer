import logging
import json
import re
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import string

from backend.config import get_config

# Configure logging
logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Types of prompts available"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    INSTRUCTION = "instruction"
    EXAMPLE = "example"
    CONTEXT = "context"

class AgentRole(Enum):
    """Agent roles for role-specific prompts"""
    RESEARCHER = "researcher"
    DRAFTER = "drafter"
    SUMMARIZER = "summarizer"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    GENERAL = "general"

class LegalDomain(Enum):
    """Legal practice areas for domain-specific prompts"""
    GENERAL = "general"
    CONTRACT = "contract"
    TORT = "tort"
    CRIMINAL = "criminal"
    CONSTITUTIONAL = "constitutional"
    CORPORATE = "corporate"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    EMPLOYMENT = "employment"
    FAMILY = "family"
    IMMIGRATION = "immigration"
    REAL_ESTATE = "real_estate"
    BANKRUPTCY = "bankruptcy"

class PromptComplexity(Enum):
    """Complexity levels for prompts"""
    SIMPLE = "simple"          # Basic, straightforward prompts
    INTERMEDIATE = "intermediate"  # Moderately complex with some guidance
    ADVANCED = "advanced"      # Complex with detailed instructions
    EXPERT = "expert"         # Highly sophisticated for expert use

@dataclass
class PromptTemplate:
    """Template definition for generating prompts"""
    template_id: str
    name: str
    description: str
    prompt_type: PromptType
    agent_role: AgentRole
    legal_domain: LegalDomain = LegalDomain.GENERAL
    complexity: PromptComplexity = PromptComplexity.INTERMEDIATE
    
    # Template content
    template_text: str
    required_variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)
    
    # Formatting options
    use_chatml_format: bool = True
    include_examples: bool = False
    include_constraints: bool = True
    include_output_format: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "prompt_type": self.prompt_type.value,
            "agent_role": self.agent_role.value,
            "legal_domain": self.legal_domain.value,
            "complexity": self.complexity.value,
            "required_variables": self.required_variables,
            "optional_variables": self.optional_variables,
            "use_chatml_format": self.use_chatml_format,
            "version": self.version,
            "tags": self.tags
        }

@dataclass
class PromptContext:
    """Context information for prompt generation"""
    # Case information
    case_name: Optional[str] = None
    case_number: Optional[str] = None
    jurisdiction: Optional[str] = None
    court: Optional[str] = None
    
    # Client information
    client_name: Optional[str] = None
    opposing_party: Optional[str] = None
    
    # Document context
    document_type: Optional[str] = None
    urgency_level: str = "normal"
    
    # Research context
    research_focus: List[str] = field(default_factory=list)
    prior_research: List[str] = field(default_factory=list)
    
    # User preferences
    citation_style: str = "bluebook"
    formality_level: str = "formal"
    target_audience: str = "attorney"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "case_name": self.case_name,
            "case_number": self.case_number,
            "jurisdiction": self.jurisdiction,
            "court": self.court,
            "client_name": self.client_name,
            "opposing_party": self.opposing_party,
            "document_type": self.document_type,
            "urgency_level": self.urgency_level,
            "research_focus": self.research_focus,
            "citation_style": self.citation_style,
            "formality_level": self.formality_level,
            "target_audience": self.target_audience
        }

class PromptTemplateManager:
    """
    Advanced prompt template management system for legal AI agents
    Provides sophisticated template creation, customization, and dynamic generation
    """
    
    def __init__(self):
        """Initialize the Prompt Template Manager"""
        self.config = get_config()
        self.templates: Dict[str, PromptTemplate] = {}
        self.template_cache: Dict[str, str] = {}
        
        # Load built-in templates
        self._load_builtin_templates()
        
        logger.info("Prompt Template Manager initialized")
    
    def _load_builtin_templates(self):
        """Load built-in prompt templates"""
        
        # Research Templates
        self._create_research_templates()
        
        # Drafting Templates
        self._create_drafting_templates()
        
        # Summarization Templates
        self._create_summarization_templates()
        
        # Analysis Templates
        self._create_analysis_templates()
        
        # Review Templates
        self._create_review_templates()
        
        logger.info(f"Loaded {len(self.templates)} built-in templates")
    
    def _create_research_templates(self):
        """Create research-specific prompt templates"""
        
        # Basic Legal Research Template
        basic_research = PromptTemplate(
            template_id="research_basic",
            name="Basic Legal Research",
            description="General legal research prompt for comprehensive analysis",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.RESEARCHER,
            complexity=PromptComplexity.INTERMEDIATE,
            template_text="""<|im_start|>system
You are an expert legal research assistant with extensive knowledge of law and legal precedent. Your role is to conduct thorough, accurate legal research and provide comprehensive analysis.

Research Guidelines:
- Provide accurate, well-sourced legal information and analysis
- Focus on relevant case law, statutes, and legal precedents
- Maintain objectivity and cite your reasoning
- Consider multiple jurisdictions when relevant
- Identify potential counterarguments and alternative interpretations

{research_constraints}

{jurisdiction_guidance}

{citation_requirements}
<|im_end|>""",
            required_variables=["query"],
            optional_variables=["jurisdiction", "focus_areas", "time_period", "legal_domain"],
            tags=["research", "general", "comprehensive"]
        )
        
        # Case Law Research Template
        case_law_research = PromptTemplate(
            template_id="research_case_law",
            name="Case Law Research",
            description="Focused case law research and precedent analysis",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.RESEARCHER,
            complexity=PromptComplexity.ADVANCED,
            template_text="""<|im_start|>system
You are a specialized case law researcher with expertise in judicial precedent and legal reasoning. Focus on identifying, analyzing, and synthesizing relevant case law.

Case Law Research Methodology:
1. Identify controlling precedent in the relevant jurisdiction
2. Analyze factual similarities and distinctions
3. Examine the reasoning and holding of key cases
4. Consider the hierarchical authority of courts
5. Identify trends in judicial interpretation
6. Note any overruled or distinguished cases

{case_analysis_framework}

{precedent_hierarchy}

Output Format:
- Key Cases: [List most relevant cases with citations]
- Controlling Precedent: [Identify binding authority]
- Persuasive Authority: [List influential but non-binding cases]
- Legal Analysis: [Synthesize the legal principles]
- Factual Distinctions: [Note important factual differences]
<|im_end|>""",
            required_variables=["legal_issue", "jurisdiction"],
            optional_variables=["case_facts", "court_level", "time_period"],
            tags=["research", "case_law", "precedent", "advanced"]
        )
        
        # Statutory Research Template
        statutory_research = PromptTemplate(
            template_id="research_statutory",
            name="Statutory Research",
            description="Comprehensive statutory and regulatory research",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.RESEARCHER,
            complexity=PromptComplexity.INTERMEDIATE,
            template_text="""<|im_start|>system
You are a statutory research specialist focused on legislation, regulations, and administrative law. Provide comprehensive analysis of relevant statutes and regulations.

Statutory Research Approach:
1. Identify applicable statutes and regulations
2. Analyze statutory language and legislative intent
3. Review regulatory interpretations and guidance
4. Consider enforcement history and agency positions
5. Examine recent amendments and proposed changes

{statutory_analysis_framework}

{regulatory_hierarchy}

Focus Areas:
- Primary Statutes: [Identify controlling legislation]
- Implementing Regulations: [Review agency rules]
- Legislative History: [Consider intent and purpose]
- Agency Guidance: [Include interpretive materials]
- Recent Developments: [Note changes and trends]
<|im_end|>""",
            required_variables=["legal_area", "jurisdiction"],
            optional_variables=["specific_statute", "agency", "effective_date"],
            tags=["research", "statutory", "regulatory", "legislation"]
        )
        
        self.templates.update({
            "research_basic": basic_research,
            "research_case_law": case_law_research,
            "research_statutory": statutory_research
        })
    
    def _create_drafting_templates(self):
        """Create document drafting prompt templates"""
        
        # Motion Drafting Template
        motion_drafting = PromptTemplate(
            template_id="draft_motion",
            name="Legal Motion Drafting",
            description="Professional legal motion drafting with proper structure",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.DRAFTER,
            complexity=PromptComplexity.ADVANCED,
            template_text="""<|im_start|>system
You are an expert legal motion writer with extensive experience in civil litigation. Draft professional, persuasive motions that comply with court rules and legal standards.

Motion Drafting Standards:
- Use proper legal citation format ({citation_style})
- Maintain professional, persuasive tone
- Follow jurisdiction-specific formatting requirements
- Include comprehensive legal analysis
- Address potential counterarguments
- Ensure factual accuracy and legal precision

Document Structure:
1. Caption and Title
2. Introduction and Relief Sought
3. Factual Background
4. Legal Standard
5. Argument Section
6. Conclusion
7. Certificate of Service

{motion_specific_guidance}

{court_rules_reminder}

Legal Writing Guidelines:
- Use clear, concise language
- Support all arguments with authority
- Organize arguments logically
- Use headings and subheadings
- Include proper citations throughout
<|im_end|>""",
            required_variables=["motion_type", "legal_basis", "relief_sought"],
            optional_variables=["facts", "opposing_arguments", "supporting_cases", "jurisdiction_rules"],
            tags=["drafting", "motion", "litigation", "professional"]
        )
        
        # Brief Drafting Template
        brief_drafting = PromptTemplate(
            template_id="draft_brief",
            name="Legal Brief Drafting",
            description="Comprehensive legal brief writing for appellate and trial courts",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.DRAFTER,
            complexity=PromptComplexity.EXPERT,
            template_text="""<|im_start|>system
You are a master brief writer with expertise in appellate advocacy and legal argumentation. Create compelling, well-reasoned legal briefs that persuasively present your client's position.

Brief Writing Excellence:
- Craft a compelling narrative that weaves facts and law
- Use the CRAC method (Conclusion, Rule, Application, Conclusion)
- Employ sophisticated legal reasoning and analysis
- Address weaknesses honestly and strategically
- Use persuasive but measured language
- Maintain credibility through accurate citations

Brief Structure:
1. Table of Contents
2. Table of Authorities  
3. Statement of Issues
4. Statement of the Case
5. Summary of Argument
6. Argument Section
7. Conclusion

{brief_specific_requirements}

{appellate_standards}

Advanced Techniques:
- Lead with your strongest argument
- Use topic sentences effectively
- Employ parallel structure
- Balance emotion with logic
- Anticipate and address counterarguments
- Use footnotes strategically
<|im_end|>""",
            required_variables=["brief_type", "legal_issues", "client_position"],
            optional_variables=["case_facts", "procedural_history", "standard_of_review", "opposing_arguments"],
            tags=["drafting", "brief", "appellate", "expert", "advocacy"]
        )
        
        # Contract Drafting Template
        contract_drafting = PromptTemplate(
            template_id="draft_contract",
            name="Contract Drafting",
            description="Professional contract and agreement drafting",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.DRAFTER,
            legal_domain=LegalDomain.CONTRACT,
            complexity=PromptComplexity.ADVANCED,
            template_text="""<|im_start|>system
You are a skilled contract attorney specializing in commercial agreements. Draft clear, comprehensive contracts that protect your client's interests while facilitating successful business relationships.

Contract Drafting Principles:
- Use precise, unambiguous language
- Address all material terms and conditions
- Include appropriate risk allocation mechanisms
- Provide clear enforcement mechanisms
- Consider applicable law and jurisdiction
- Build in flexibility for changed circumstances

Essential Contract Elements:
1. Parties and Capacity
2. Consideration and Terms
3. Performance Obligations
4. Risk Allocation
5. Dispute Resolution
6. Termination Provisions
7. Miscellaneous Provisions

{contract_specific_guidance}

{risk_considerations}

Drafting Best Practices:
- Define all key terms clearly
- Use consistent terminology throughout
- Avoid ambiguous pronouns
- Include appropriate representations and warranties
- Address force majeure and other contingencies
- Consider tax and regulatory implications
<|im_end|>""",
            required_variables=["contract_type", "parties", "key_terms"],
            optional_variables=["governing_law", "dispute_resolution", "special_provisions", "industry_standards"],
            tags=["drafting", "contract", "commercial", "transactional"]
        )
        
        self.templates.update({
            "draft_motion": motion_drafting,
            "draft_brief": brief_drafting,
            "draft_contract": contract_drafting
        })
    
    def _create_summarization_templates(self):
        """Create summarization prompt templates"""
        
        # Executive Summary Template
        executive_summary = PromptTemplate(
            template_id="summary_executive",
            name="Executive Summary",
            description="High-level executive summary for legal documents",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.SUMMARIZER,
            complexity=PromptComplexity.INTERMEDIATE,
            template_text="""<|im_start|>system
You are an expert legal summarizer specializing in creating clear, concise executive summaries for legal professionals and clients. Focus on key insights, actionable recommendations, and strategic implications.

Executive Summary Standards:
- Lead with the most important information
- Use clear, accessible language appropriate for {target_audience}
- Highlight key risks and opportunities
- Provide actionable recommendations
- Maintain accuracy while ensuring readability
- Structure information logically and scannable

Summary Structure:
1. Key Findings (2-3 bullet points)
2. Critical Issues and Risks
3. Recommendations and Next Steps
4. Strategic Implications
5. Timeline and Priorities

{audience_specific_guidance}

{format_preferences}

Writing Guidelines:
- Use active voice and strong verbs
- Avoid unnecessary legal jargon
- Include specific examples when helpful
- Quantify impacts when possible
- Maintain professional but accessible tone
<|im_end|>""",
            required_variables=["content_to_summarize", "target_audience"],
            optional_variables=["summary_length", "focus_areas", "urgency_level", "client_context"],
            tags=["summary", "executive", "strategic", "accessible"]
        )
        
        # Research Summary Template
        research_summary = PromptTemplate(
            template_id="summary_research",
            name="Research Summary",
            description="Comprehensive research findings summary",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.SUMMARIZER,
            complexity=PromptComplexity.ADVANCED,
            template_text="""<|im_start|>system
You are a legal research summarizer with expertise in distilling complex legal research into clear, actionable insights. Preserve important legal nuances while making content accessible.

Research Summary Framework:
- Synthesize multiple sources into coherent analysis
- Preserve critical legal distinctions and nuances
- Identify patterns and trends in the law
- Highlight conflicting authorities or interpretations
- Assess the strength and reliability of sources
- Connect research to practical implications

Summary Components:
1. Research Overview
2. Key Legal Principles
3. Controlling Authority
4. Conflicting Interpretations
5. Practical Applications
6. Research Gaps and Limitations

{research_synthesis_guidance}

{authority_hierarchy}

Quality Standards:
- Maintain legal accuracy and precision
- Use proper legal citations
- Clearly distinguish between different types of authority
- Note any limitations or qualifications
- Identify areas requiring additional research
<|im_end|>""",
            required_variables=["research_results", "legal_question"],
            optional_variables=["jurisdiction_focus", "practice_implications", "client_specific_issues"],
            tags=["summary", "research", "synthesis", "analysis"]
        )
        
        self.templates.update({
            "summary_executive": executive_summary,
            "summary_research": research_summary
        })
    
    def _create_analysis_templates(self):
        """Create legal analysis prompt templates"""
        
        # Issue Analysis Template
        issue_analysis = PromptTemplate(
            template_id="analysis_issue",
            name="Legal Issue Analysis",
            description="Comprehensive legal issue analysis and assessment",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.ANALYST,
            complexity=PromptComplexity.ADVANCED,
            template_text="""<|im_start|>system
You are a senior legal analyst specializing in complex legal issue assessment. Provide thorough, balanced analysis that considers multiple perspectives and potential outcomes.

Analysis Methodology:
- Apply relevant legal framework systematically
- Consider factual variations and their legal implications
- Assess strengths and weaknesses of different positions
- Evaluate procedural and substantive considerations
- Identify key uncertainties and risk factors
- Provide probability assessments where appropriate

Analytical Framework:
1. Issue Identification and Framing
2. Applicable Legal Standards
3. Factual Analysis and Application
4. Strength Assessment
5. Risk Evaluation
6. Strategic Considerations

{analysis_specific_guidance}

{risk_assessment_framework}

Professional Standards:
- Maintain analytical objectivity
- Support conclusions with reasoned analysis
- Acknowledge limitations and uncertainties
- Consider opposing viewpoints fairly
- Provide actionable insights
<|im_end|>""",
            required_variables=["legal_issue", "relevant_facts"],
            optional_variables=["client_objectives", "risk_tolerance", "strategic_context", "time_constraints"],
            tags=["analysis", "assessment", "strategic", "comprehensive"]
        )
        
        self.templates.update({
            "analysis_issue": issue_analysis
        })
    
    def _create_review_templates(self):
        """Create document review prompt templates"""
        
        # Document Review Template
        document_review = PromptTemplate(
            template_id="review_document",
            name="Legal Document Review",
            description="Comprehensive legal document review and analysis",
            prompt_type=PromptType.SYSTEM,
            agent_role=AgentRole.REVIEWER,
            complexity=PromptComplexity.EXPERT,
            template_text="""<|im_start|>system
You are an expert legal document reviewer with extensive experience in identifying issues, risks, and improvement opportunities in legal documents.

Review Methodology:
- Conduct systematic analysis of legal and business terms
- Identify potential risks and liabilities
- Assess compliance with applicable law and standards
- Evaluate clarity and enforceability of provisions
- Consider practical implementation challenges
- Recommend specific improvements and alternatives

Review Categories:
1. Legal Compliance and Validity
2. Risk Assessment and Allocation
3. Clarity and Enforceability
4. Business Terms and Fairness
5. Implementation Considerations
6. Recommended Revisions

{review_specific_guidance}

{risk_categories}

Quality Assurance:
- Provide specific, actionable feedback
- Prioritize issues by significance and urgency
- Explain the reasoning behind recommendations
- Consider client's business objectives
- Maintain professional and constructive tone
<|im_end|>""",
            required_variables=["document_content", "review_purpose"],
            optional_variables=["client_priorities", "risk_tolerance", "industry_context", "regulatory_requirements"],
            tags=["review", "analysis", "risk", "compliance", "expert"]
        )
        
        self.templates.update({
            "review_document": document_review
        })
    
    def register_template(self, template: PromptTemplate) -> bool:
        """
        Register a new prompt template
        
        Args:
            template: PromptTemplate to register
            
        Returns:
            bool: True if registration successful
        """
        
        if template.template_id in self.templates:
            logger.warning(f"Template {template.template_id} already exists, updating...")
        
        self.templates[template.template_id] = template
        
        # Clear cache for this template
        cache_keys_to_remove = [key for key in self.template_cache.keys() 
                               if key.startswith(template.template_id)]
        for key in cache_keys_to_remove:
            del self.template_cache[key]
        
        logger.info(f"Template registered: {template.template_id}")
        return True
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self,
                      agent_role: Optional[AgentRole] = None,
                      legal_domain: Optional[LegalDomain] = None,
                      complexity: Optional[PromptComplexity] = None,
                      tags: Optional[List[str]] = None) -> List[PromptTemplate]:
        """
        List templates with optional filtering
        
        Args:
            agent_role: Filter by agent role
            legal_domain: Filter by legal domain
            complexity: Filter by complexity level
            tags: Filter by tags (any match)
            
        Returns:
            List[PromptTemplate]: Matching templates
        """
        
        templates = list(self.templates.values())
        
        if agent_role:
            templates = [t for t in templates if t.agent_role == agent_role]
        
        if legal_domain:
            templates = [t for t in templates if t.legal_domain == legal_domain]
        
        if complexity:
            templates = [t for t in templates if t.complexity == complexity]
        
        if tags:
            templates = [t for t in templates 
                        if any(tag in t.tags for tag in tags)]
        
        return templates
    
    def generate_prompt(self,
                       template_id: str,
                       variables: Dict[str, Any],
                       context: Optional[PromptContext] = None) -> str:
        """
        Generate a prompt from a template with variable substitution
        
        Args:
            template_id: ID of template to use
            variables: Variables to substitute in template
            context: Additional context information
            
        Returns:
            str: Generated prompt
        """
        
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Check for required variables
        missing_vars = set(template.required_variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Generate cache key
        cache_key = f"{template_id}_{hash(str(sorted(variables.items())))}"
        if context:
            cache_key += f"_{hash(str(sorted(context.to_dict().items())))}"
        
        # Check cache
        if cache_key in self.template_cache:
            logger.debug(f"Using cached prompt for {template_id}")
            return self.template_cache[cache_key]
        
        # Prepare all variables for substitution
        all_variables = variables.copy()
        
        # Add context variables
        if context:
            all_variables.update(context.to_dict())
        
        # Add dynamic variables
        all_variables.update(self._get_dynamic_variables(template, context))
        
        # Perform variable substitution
        try:
            prompt = self._substitute_variables(template.template_text, all_variables)
            
            # Apply post-processing
            prompt = self._apply_post_processing(prompt, template, context)
            
            # Cache the result
            self.template_cache[cache_key] = prompt
            
            logger.debug(f"Generated prompt for template {template_id}")
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to generate prompt for {template_id}: {e}")
            raise
    
    def _get_dynamic_variables(self,
                              template: PromptTemplate,
                              context: Optional[PromptContext]) -> Dict[str, str]:
        """Generate dynamic variables based on template and context"""
        
        dynamic_vars = {}
        
        # Citation requirements
        if context and context.citation_style:
            if context.citation_style.lower() == "bluebook":
                dynamic_vars["citation_requirements"] = """
Citation Requirements:
- Use Bluebook citation format (20th edition)
- Include pinpoint citations for specific propositions
- Use proper short form citations for subsequent references
- Include parentheticals when helpful for clarity
"""
            else:
                dynamic_vars["citation_requirements"] = f"""
Citation Requirements:
- Use {context.citation_style} citation format
- Include specific page or section references
- Maintain consistency throughout the document
"""
        else:
            dynamic_vars["citation_requirements"] = """
Citation Requirements:
- Use standard legal citation format
- Include specific references where possible
- Ensure accuracy and consistency
"""
        
        # Jurisdiction guidance
        if context and context.jurisdiction:
            dynamic_vars["jurisdiction_guidance"] = f"""
Jurisdictional Focus:
- Primary jurisdiction: {context.jurisdiction}
- Apply {context.jurisdiction} law and precedent
- Note when federal law preempts state law
- Consider jurisdictional variations when relevant
"""
        else:
            dynamic_vars["jurisdiction_guidance"] = """
Jurisdictional Considerations:
- Consider applicable federal and state law
- Note jurisdictional variations when relevant
- Identify controlling vs. persuasive authority
"""
        
        # Research constraints
        if template.agent_role == AgentRole.RESEARCHER:
            dynamic_vars["research_constraints"] = """
Research Constraints:
- Focus on authoritative sources (cases, statutes, regulations)
- Distinguish between primary and secondary authority
- Consider the current state of the law
- Note any pending legislation or recent developments
"""
        
        # Formality level
        if context and context.formality_level:
            if context.formality_level == "formal":
                dynamic_vars["tone_guidance"] = "Maintain formal, professional legal language throughout."
            elif context.formality_level == "business":
                dynamic_vars["tone_guidance"] = "Use business-professional language that is accessible but precise."
            else:
                dynamic_vars["tone_guidance"] = "Use clear, accessible language while maintaining legal accuracy."
        
        # Target audience adaptations
        if context and context.target_audience:
            if context.target_audience == "judge":
                dynamic_vars["audience_specific_guidance"] = """
Judicial Audience Considerations:
- Use respectful, deferential tone
- Focus on legal reasoning and precedent
- Be concise and well-organized
- Avoid hyperbole or emotional language
"""
            elif context.target_audience == "client":
                dynamic_vars["audience_specific_guidance"] = """
Client Communication Guidelines:
- Explain legal concepts clearly
- Focus on practical implications
- Highlight risks and opportunities
- Provide actionable recommendations
"""
        
        # Document type specific guidance
        if context and context.document_type:
            if context.document_type == "motion":
                dynamic_vars["motion_specific_guidance"] = """
Motion-Specific Requirements:
- Begin with clear statement of relief sought
- Include proper legal standard for the motion type
- Address all elements required for relief
- Anticipate and respond to likely objections
"""
        
        return dynamic_vars
    
    def _substitute_variables(self, template_text: str, variables: Dict[str, Any]) -> str:
        """Perform variable substitution in template text"""
        
        # Use string Template for safe substitution
        template = string.Template(template_text)
        
        try:
            # Perform substitution
            result = template.safe_substitute(variables)
            
            # Clean up any remaining empty variable placeholders
            result = re.sub(r'\{\w+\}\s*\n', '', result)  # Remove empty lines with variables
            result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)  # Clean up multiple blank lines
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Variable substitution failed: {e}")
            raise
    
    def _apply_post_processing(self,
                              prompt: str,
                              template: PromptTemplate,
                              context: Optional[PromptContext]) -> str:
        """Apply post-processing to the generated prompt"""
        
        # Ensure proper ChatML formatting if required
        if template.use_chatml_format and not prompt.startswith("<|im_start|>"):
            prompt = f"<|im_start|>system\n{prompt}\n<|im_end|>"
        
        # Add examples if requested and available
        if template.include_examples:
            examples = self._get_examples_for_template(template)
            if examples:
                prompt += f"\n\nExamples:\n{examples}"
        
        # Add output format guidance if requested
        if template.include_output_format:
            format_guidance = self._get_output_format_guidance(template)
            if format_guidance:
                prompt += f"\n\n{format_guidance}"
        
        return prompt
    
    def _get_examples_for_template(self, template: PromptTemplate) -> str:
        """Get examples for a specific template"""
        
        # This would typically load from a database or file
        # For now, return empty string
        return ""
    
    def _get_output_format_guidance(self, template: PromptTemplate) -> str:
        """Get output format guidance for a specific template"""
        
        format_guidance = {
            AgentRole.RESEARCHER: """
Output Format:
- Provide a structured response with clear headings
- Include a summary of key findings
- List relevant authorities with proper citations
- Note any limitations or areas requiring further research
- Conclude with practical implications
""",
            
            AgentRole.DRAFTER: """
Output Format:
- Follow standard legal document structure
- Use proper headings and numbering
- Include table of contents for longer documents
- Maintain consistent formatting throughout
- End with appropriate signature block
""",
            
            AgentRole.SUMMARIZER: """
Output Format:
- Begin with executive summary (2-3 sentences)
- Use bullet points for key findings
- Organize information by importance
- Include specific recommendations
- End with next steps or action items
""",
            
            AgentRole.ANALYST: """
Output Format:
- Present analysis in logical order
- Use clear section headings
- Support conclusions with specific evidence
- Include risk assessment with probability estimates
- Provide actionable recommendations
""",
            
            AgentRole.REVIEWER: """
Output Format:
- Categorize findings by severity/importance
- Provide specific line-by-line comments where relevant
- Include summary of major issues
- Recommend specific revisions
- Prioritize action items
"""
        }
        
        return format_guidance.get(template.agent_role, "")
    
    def create_custom_template(self,
                              template_id: str,
                              name: str,
                              agent_role: AgentRole,
                              template_text: str,
                              **kwargs) -> PromptTemplate:
        """
        Create a custom prompt template
        
        Args:
            template_id: Unique identifier
            name: Human-readable name
            agent_role: Target agent role
            template_text: Template content with variables
            **kwargs: Additional template options
            
        Returns:
            PromptTemplate: Created template
        """
        
        # Extract required variables from template text
        required_vars = re.findall(r'\{(\w+)\}', template_text)
        required_vars = list(set(required_vars))  # Remove duplicates
        
        template = PromptTemplate(
            template_id=template_id,
            name=name,
            description=kwargs.get('description', f"Custom template for {agent_role.value}"),
            prompt_type=kwargs.get('prompt_type', PromptType.SYSTEM),
            agent_role=agent_role,
            legal_domain=kwargs.get('legal_domain', LegalDomain.GENERAL),
            complexity=kwargs.get('complexity', PromptComplexity.INTERMEDIATE),
            template_text=template_text,
            required_variables=kwargs.get('required_variables', required_vars),
            optional_variables=kwargs.get('optional_variables', []),
            use_chatml_format=kwargs.get('use_chatml_format', True),
            include_examples=kwargs.get('include_examples', False),
            include_constraints=kwargs.get('include_constraints', True),
            include_output_format=kwargs.get('include_output_format', True),
            tags=kwargs.get('tags', ['custom'])
        )
        
        # Register the template
        self.register_template(template)
        
        logger.info(f"Created custom template: {template_id}")
        return template
    
    def clone_template(self,
                      source_template_id: str,
                      new_template_id: str,
                      modifications: Dict[str, Any] = None) -> PromptTemplate:
        """
        Clone an existing template with optional modifications
        
        Args:
            source_template_id: ID of template to clone
            new_template_id: ID for new template
            modifications: Optional modifications to apply
            
        Returns:
            PromptTemplate: Cloned template
        """
        
        source_template = self.get_template(source_template_id)
        if not source_template:
            raise ValueError(f"Source template {source_template_id} not found")
        
        # Create copy of source template
        new_template = PromptTemplate(
            template_id=new_template_id,
            name=source_template.name + " (Clone)",
            description=source_template.description,
            prompt_type=source_template.prompt_type,
            agent_role=source_template.agent_role,
            legal_domain=source_template.legal_domain,
            complexity=source_template.complexity,
            template_text=source_template.template_text,
            required_variables=source_template.required_variables.copy(),
            optional_variables=source_template.optional_variables.copy(),
            use_chatml_format=source_template.use_chatml_format,
            include_examples=source_template.include_examples,
            include_constraints=source_template.include_constraints,
            include_output_format=source_template.include_output_format,
            tags=source_template.tags.copy() + ["cloned"]
        )
        
        # Apply modifications
        if modifications:
            for key, value in modifications.items():
                if hasattr(new_template, key):
                    setattr(new_template, key, value)
        
        # Register the cloned template
        self.register_template(new_template)
        
        logger.info(f"Cloned template {source_template_id} to {new_template_id}")
        return new_template
    
    def validate_template(self, template: PromptTemplate) -> Dict[str, Any]:
        """
        Validate a template for completeness and correctness
        
        Args:
            template: Template to validate
            
        Returns:
            Dict containing validation results
        """
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check required fields
        if not template.template_id:
            validation_result["errors"].append("Template ID is required")
            validation_result["valid"] = False
        
        if not template.name:
            validation_result["errors"].append("Template name is required")
            validation_result["valid"] = False
        
        if not template.template_text:
            validation_result["errors"].append("Template text is required")
            validation_result["valid"] = False
        
        # Check template text for issues
        if template.template_text:
            # Check for unmatched braces
            open_braces = template.template_text.count('{')
            close_braces = template.template_text.count('}')
            if open_braces != close_braces:
                validation_result["errors"].append("Unmatched braces in template text")
                validation_result["valid"] = False
            
            # Check for required variables in text
            template_vars = set(re.findall(r'\{(\w+)\}', template.template_text))
            declared_vars = set(template.required_variables + template.optional_variables)
            
            undeclared_vars = template_vars - declared_vars
            if undeclared_vars:
                validation_result["warnings"].append(f"Undeclared variables in template: {undeclared_vars}")
            
            unused_vars = declared_vars - template_vars
            if unused_vars:
                validation_result["warnings"].append(f"Declared but unused variables: {unused_vars}")
        
        # Check ChatML format
        if template.use_chatml_format and template.template_text:
            if not ("<|im_start|>" in template.template_text and "<|im_end|>" in template.template_text):
                validation_result["suggestions"].append("Consider adding proper ChatML formatting")
        
        # Check complexity appropriateness
        if template.complexity == PromptComplexity.SIMPLE and len(template.template_text) > 500:
            validation_result["warnings"].append("Template marked as simple but content is lengthy")
        
        return validation_result
    
    def export_templates(self, template_ids: List[str] = None) -> Dict[str, Any]:
        """
        Export templates to dictionary format
        
        Args:
            template_ids: Optional list of specific template IDs to export
            
        Returns:
            Dict containing exported templates
        """
        
        if template_ids:
            templates_to_export = {tid: self.templates[tid] for tid in template_ids if tid in self.templates}
        else:
            templates_to_export = self.templates
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "template_count": len(templates_to_export),
            "templates": {tid: template.to_dict() for tid, template in templates_to_export.items()}
        }
        
        logger.info(f"Exported {len(templates_to_export)} templates")
        return export_data
    
    def import_templates(self, import_data: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        """
        Import templates from dictionary format
        
        Args:
            import_data: Dictionary containing template data
            overwrite: Whether to overwrite existing templates
            
        Returns:
            Dict containing import results
        """
        
        import_result = {
            "imported": 0,
            "skipped": 0,
            "errors": []
        }
        
        if "templates" not in import_data:
            import_result["errors"].append("No templates found in import data")
            return import_result
        
        for template_id, template_data in import_data["templates"].items():
            try:
                # Check if template already exists
                if template_id in self.templates and not overwrite:
                    import_result["skipped"] += 1
                    continue
                
                # Create template from data
                template = PromptTemplate(
                    template_id=template_data["template_id"],
                    name=template_data["name"],
                    description=template_data["description"],
                    prompt_type=PromptType(template_data["prompt_type"]),
                    agent_role=AgentRole(template_data["agent_role"]),
                    legal_domain=LegalDomain(template_data.get("legal_domain", "general")),
                    complexity=PromptComplexity(template_data.get("complexity", "intermediate")),
                    template_text=template_data.get("template_text", ""),
                    required_variables=template_data.get("required_variables", []),
                    optional_variables=template_data.get("optional_variables", []),
                    use_chatml_format=template_data.get("use_chatml_format", True),
                    version=template_data.get("version", "1.0"),
                    tags=template_data.get("tags", [])
                )
                
                # Validate template
                validation = self.validate_template(template)
                if not validation["valid"]:
                    import_result["errors"].append(f"Template {template_id} validation failed: {validation['errors']}")
                    continue
                
                # Register template
                self.register_template(template)
                import_result["imported"] += 1
                
            except Exception as e:
                import_result["errors"].append(f"Failed to import template {template_id}: {str(e)}")
        
        logger.info(f"Import completed: {import_result['imported']} imported, {import_result['skipped']} skipped")
        return import_result
    
    def get_template_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for templates"""
        
        # This would typically track actual usage from logs/database
        # For now, return basic statistics
        
        stats = {
            "total_templates": len(self.templates),
            "by_agent_role": {},
            "by_legal_domain": {},
            "by_complexity": {},
            "cache_hits": len(self.template_cache),
            "most_used_templates": []  # Would be populated from usage logs
        }
        
        # Count by categories
        for template in self.templates.values():
            # By agent role
            role = template.agent_role.value
            stats["by_agent_role"][role] = stats["by_agent_role"].get(role, 0) + 1
            
            # By legal domain
            domain = template.legal_domain.value
            stats["by_legal_domain"][domain] = stats["by_legal_domain"].get(domain, 0) + 1
            
            # By complexity
            complexity = template.complexity.value
            stats["by_complexity"][complexity] = stats["by_complexity"].get(complexity, 0) + 1
        
        return stats
    
    def clear_cache(self):
        """Clear the template cache"""
        self.template_cache.clear()
        logger.info("Template cache cleared")

# Global template manager instance
_template_manager: Optional[PromptTemplateManager] = None

def get_template_manager() -> PromptTemplateManager:
    """Get or create the global prompt template manager instance"""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    return _template_manager

# Convenience functions for common operations
def generate_research_prompt(query: str, **kwargs) -> str:
    """Generate a research prompt using the basic research template"""
    manager = get_template_manager()
    return manager.generate_prompt("research_basic", {"query": query}, **kwargs)

def generate_drafting_prompt(document_type: str, **kwargs) -> str:
    """Generate a drafting prompt based on document type"""
    manager = get_template_manager()
    
    template_map = {
        "motion": "draft_motion",
        "brief": "draft_brief", 
        "contract": "draft_contract"
    }
    
    template_id = template_map.get(document_type.lower(), "draft_motion")
    return manager.generate_prompt(template_id, kwargs)

def generate_summary_prompt(content_type: str, **kwargs) -> str:
    """Generate a summary prompt based on content type"""
    manager = get_template_manager()
    
    template_map = {
        "executive": "summary_executive",
        "research": "summary_research"
    }
    
    template_id = template_map.get(content_type.lower(), "summary_executive")
    return manager.generate_prompt(template_id, kwargs)

# Template creation helpers
def create_quick_template(template_id: str, 
                         agent_role: str, 
                         template_text: str,
                         **kwargs) -> PromptTemplate:
    """Quick template creation helper"""
    manager = get_template_manager()
    role = AgentRole(agent_role.lower())
    return manager.create_custom_template(template_id, template_id.replace("_", " ").title(), role, template_text, **kwargs)