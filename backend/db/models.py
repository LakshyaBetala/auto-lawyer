"""
AutoLawyer Database Models
SQLAlchemy ORM models for the legal brief drafting system
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

class BriefStatus(enum.Enum):
    """Status enumeration for legal briefs"""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress" 
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class AgentRole(enum.Enum):
    """Available agent roles in the system"""
    RESEARCHER = "researcher"
    DRAFTER = "drafter"
    SUMMARIZER = "summarizer"
    REVIEWER = "reviewer"

class User(Base):
    """User model for authentication and brief ownership"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    briefs = relationship("Brief", back_populates="owner")
    agent_logs = relationship("AgentLog", back_populates="user")

class Brief(Base):
    """Main legal brief model"""
    __tablename__ = "briefs"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    case_type = Column(String(100))  # e.g., "civil", "criminal", "contract"
    jurisdiction = Column(String(100))  # e.g., "federal", "state", "local"
    court_level = Column(String(50))  # e.g., "trial", "appellate", "supreme"
    
    # Brief content and status
    content = Column(Text)  # Main brief content
    research_notes = Column(Text)  # Research findings
    citations = Column(JSON)  # Array of legal citations
    status = Column(Enum(BriefStatus), default=BriefStatus.DRAFT)
    
    # Metadata
    word_count = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    estimated_completion = Column(DateTime)
    
    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime)
    
    # Relationships
    owner = relationship("User", back_populates="briefs")
    versions = relationship("BriefVersion", back_populates="brief", cascade="all, delete-orphan")
    agent_logs = relationship("AgentLog", back_populates="brief", cascade="all, delete-orphan")
    research_results = relationship("ResearchResult", back_populates="brief", cascade="all, delete-orphan")

class BriefVersion(Base):
    """Version control for brief drafts"""
    __tablename__ = "brief_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    brief_id = Column(Integer, ForeignKey("briefs.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    title = Column(String(200))
    content = Column(Text, nullable=False)
    changes_summary = Column(Text)  # Summary of what changed
    created_by_agent = Column(Enum(AgentRole))  # Which agent created this version
    
    # Metadata
    word_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    brief = relationship("Brief", back_populates="versions")

class AgentLog(Base):
    """Logs of agent activities and contributions"""
    __tablename__ = "agent_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    brief_id = Column(Integer, ForeignKey("briefs.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Agent information
    agent_role = Column(Enum(AgentRole), nullable=False)
    agent_name = Column(String(100))  # Custom name for agent instance
    
    # Activity details
    action = Column(String(100), nullable=False)  # e.g., "research", "draft", "summarize"
    input_prompt = Column(Text)  # Prompt sent to the agent
    output_content = Column(Text)  # Agent's response
    
    # Execution metadata
    execution_time_ms = Column(Integer)  # How long the agent took
    tokens_used = Column(Integer)  # Token count for local LLM
    model_used = Column(String(100))  # Which model was used
    success = Column(Boolean, default=True)
    error_message = Column(Text)  # If agent failed
    
    # Context and workflow
    parent_log_id = Column(Integer, ForeignKey("agent_logs.id"))  # For agent chains
    workflow_step = Column(String(100))  # Step in the workflow graph
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime)
    
    # Relationships
    brief = relationship("Brief", back_populates="agent_logs")
    user = relationship("User", back_populates="agent_logs")
    parent_log = relationship("AgentLog", remote_side=[id])

class ResearchResult(Base):
    """Storage for legal research findings"""
    __tablename__ = "research_results"
    
    id = Column(Integer, primary_key=True, index=True)
    brief_id = Column(Integer, ForeignKey("briefs.id"), nullable=False)
    
    # Research query and results
    query = Column(Text, nullable=False)  # Original research query
    source_type = Column(String(50))  # e.g., "case_law", "statute", "regulation"
    title = Column(String(300))
    citation = Column(String(500))  # Full legal citation
    court = Column(String(100))  # Court that decided the case
    year = Column(Integer)  # Year of decision
    
    # Content
    summary = Column(Text)  # AI-generated summary
    relevant_excerpt = Column(Text)  # Key passages
    full_text = Column(Text)  # Complete text if available
    
    # Relevance scoring
    relevance_score = Column(Integer)  # 1-10 relevance to the brief
    agent_notes = Column(Text)  # Agent's analysis
    
    # Metadata
    found_by_agent = Column(Enum(AgentRole), default=AgentRole.RESEARCHER)
    verified = Column(Boolean, default=False)  # Human verified
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    brief = relationship("Brief", back_populates="research_results")

class ModelConfig(Base):
    """Configuration for local LLM models"""
    __tablename__ = "model_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    model_path = Column(String(500), nullable=False)  # Path to .gguf file
    model_type = Column(String(50))  # e.g., "llama", "mistral", "codellama"
    
    # Model parameters
    context_length = Column(Integer, default=4096)
    max_tokens = Column(Integer, default=1024)
    temperature = Column(Float, default=0.7)
    top_p = Column(Float, default=0.9)
    
    # Performance settings
    n_threads = Column(Integer, default=4)
    n_gpu_layers = Column(Integer, default=0)  # For GPU acceleration
    
    # Usage tracking
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    total_calls = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Agent assignments
    preferred_for_research = Column(Boolean, default=False)
    preferred_for_drafting = Column(Boolean, default=False)
    preferred_for_summarizing = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class WorkflowTemplate(Base):
    """Templates for agent workflows"""
    __tablename__ = "workflow_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    case_type = Column(String(100))  # Which case types this applies to
    
    # Workflow definition
    workflow_graph = Column(JSON)  # LangGraph DAG definition
    agent_sequence = Column(JSON)  # Order of agent execution
    
    # Configuration
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    estimated_duration_hours = Column(Integer)
    
    # Usage stats
    times_used = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Additional indexes for performance
from sqlalchemy import Index

# Create composite indexes for common queries
Index('idx_brief_owner_status', Brief.owner_id, Brief.status)
Index('idx_agent_log_brief_role', AgentLog.brief_id, AgentLog.agent_role)
Index('idx_research_brief_relevance', ResearchResult.brief_id, ResearchResult.relevance_score)
Index('idx_version_brief_number', BriefVersion.brief_id, BriefVersion.version_number)