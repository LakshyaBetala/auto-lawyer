"""
AutoLawyer Pydantic Schemas
Data validation and serialization schemas for FastAPI endpoints
"""

from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Enums matching the database models
class BriefStatusEnum(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    REVIEW = "review" 
    COMPLETED = "completed"
    ARCHIVED = "archived"

class AgentRoleEnum(str, Enum):
    RESEARCHER = "researcher"
    DRAFTER = "drafter"
    SUMMARIZER = "summarizer"
    REVIEWER = "reviewer"

class SourceTypeEnum(str, Enum):
    CASE_LAW = "case_law"
    STATUTE = "statute"
    REGULATION = "regulation"
    SECONDARY = "secondary"
    TREATISE = "treatise"

# Base schemas for common fields
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        str_strip_whitespace = True

class TimestampMixin(BaseModel):
    created_at: datetime
    updated_at: Optional[datetime] = None

# User Schemas
class UserBase(BaseSchema):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=100)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseSchema):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    is_active: Optional[bool] = None

class UserResponse(UserBase, TimestampMixin):
    id: int
    is_active: bool
    is_admin: bool

class UserLogin(BaseSchema):
    username: str
    password: str

# Brief Schemas
class BriefBase(BaseSchema):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    case_type: Optional[str] = Field(None, max_length=100)
    jurisdiction: Optional[str] = Field(None, max_length=100)
    court_level: Optional[str] = Field(None, max_length=50)

class BriefCreate(BriefBase):
    content: Optional[str] = None
    research_notes: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

class BriefUpdate(BaseSchema):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    case_type: Optional[str] = Field(None, max_length=100)
    jurisdiction: Optional[str] = Field(None, max_length=100)
    court_level: Optional[str] = Field(None, max_length=50)
    content: Optional[str] = None
    research_notes: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = None
    status: Optional[BriefStatusEnum] = None

class BriefResponse(BriefBase, TimestampMixin):
    id: int
    content: Optional[str] = None
    research_notes: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    status: BriefStatusEnum
    word_count: int = 0
    page_count: int = 0
    estimated_completion: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    owner_id: int

class BriefSummary(BaseSchema):
    """Lightweight brief info for listings"""
    id: int
    title: str
    case_type: Optional[str] = None
    status: BriefStatusEnum
    word_count: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None

# Brief Version Schemas
class BriefVersionBase(BaseSchema):
    title: Optional[str] = Field(None, max_length=200)
    content: str = Field(..., min_length=1)
    changes_summary: Optional[str] = None
    created_by_agent: Optional[AgentRoleEnum] = None

class BriefVersionCreate(BriefVersionBase):
    brief_id: int
    version_number: int

class BriefVersionResponse(BriefVersionBase):
    id: int
    brief_id: int
    version_number: int
    word_count: int = 0
    created_at: datetime

# Agent Log Schemas
class AgentLogBase(BaseSchema):
    agent_role: AgentRoleEnum
    agent_name: Optional[str] = Field(None, max_length=100)
    action: str = Field(..., max_length=100)
    input_prompt: Optional[str] = None
    output_content: Optional[str] = None

class AgentLogCreate(AgentLogBase):
    brief_id: int
    execution_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = Field(None, max_length=100)
    success: bool = True
    error_message: Optional[str] = None
    parent_log_id: Optional[int] = None
    workflow_step: Optional[str] = Field(None, max_length=100)

class AgentLogResponse(AgentLogBase, TimestampMixin):
    id: int
    brief_id: int
    user_id: int
    execution_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    success: bool
    error_message: Optional[str] = None
    parent_log_id: Optional[int] = None
    workflow_step: Optional[str] = None
    completed_at: Optional[datetime] = None

# Research Result Schemas
class ResearchResultBase(BaseSchema):
    query: str = Field(..., min_length=1)
    source_type: SourceTypeEnum
    title: Optional[str] = Field(None, max_length=300)
    citation: Optional[str] = Field(None, max_length=500)
    court: Optional[str] = Field(None, max_length=100)
    year: Optional[int] = Field(None, ge=1800, le=2030)

class ResearchResultCreate(ResearchResultBase):
    brief_id: int
    summary: Optional[str] = None
    relevant_excerpt: Optional[str] = None
    full_text: Optional[str] = None
    relevance_score: Optional[int] = Field(None, ge=1, le=10)
    agent_notes: Optional[str] = None
    found_by_agent: AgentRoleEnum = AgentRoleEnum.RESEARCHER

class ResearchResultUpdate(BaseSchema):
    summary: Optional[str] = None
    relevant_excerpt: Optional[str] = None
    full_text: Optional[str] = None
    relevance_score: Optional[int] = Field(None, ge=1, le=10)
    agent_notes: Optional[str] = None
    verified: Optional[bool] = None

class ResearchResultResponse(ResearchResultBase):
    id: int
    brief_id: int
    summary: Optional[str] = None
    relevant_excerpt: Optional[str] = None
    relevance_score: Optional[int] = None
    agent_notes: Optional[str] = None
    found_by_agent: AgentRoleEnum
    verified: bool = False
    created_at: datetime

# Model Configuration Schemas
class ModelConfigBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=100)
    model_path: str = Field(..., min_length=1, max_length=500)
    model_type: Optional[str] = Field(None, max_length=50)
    context_length: int = Field(4096, ge=512, le=32768)
    max_tokens: int = Field(1024, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)

class ModelConfigCreate(ModelConfigBase):
    n_threads: int = Field(4, ge=1, le=32)
    n_gpu_layers: int = Field(0, ge=0, le=100)
    preferred_for_research: bool = False
    preferred_for_drafting: bool = False
    preferred_for_summarizing: bool = False

class ModelConfigUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    model_path: Optional[str] = Field(None, min_length=1, max_length=500)
    model_type: Optional[str] = Field(None, max_length=50)
    context_length: Optional[int] = Field(None, ge=512, le=32768)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    n_threads: Optional[int] = Field(None, ge=1, le=32)
    n_gpu_layers: Optional[int] = Field(None, ge=0, le=100)
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None
    preferred_for_research: Optional[bool] = None
    preferred_for_drafting: Optional[bool] = None
    preferred_for_summarizing: Optional[bool] = None

class ModelConfigResponse(ModelConfigBase, TimestampMixin):
    id: int
    n_threads: int
    n_gpu_layers: int
    is_active: bool
    is_default: bool
    total_calls: int = 0
    total_tokens: int = 0
    preferred_for_research: bool
    preferred_for_drafting: bool
    preferred_for_summarizing: bool

# Agent Execution Schemas
class AgentExecutionRequest(BaseSchema):
    brief_id: int
    agent_role: AgentRoleEnum
    prompt: str = Field(..., min_length=1)
    model_override: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)

class AgentExecutionResponse(BaseSchema):
    success: bool
    agent_role: AgentRoleEnum
    output: Optional[str] = None
    execution_time_ms: int
    tokens_used: int
    model_used: str
    error_message: Optional[str] = None
    log_id: int

# Workflow Schemas
class WorkflowTemplateBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    case_type: Optional[str] = Field(None, max_length=100)
    workflow_graph: Dict[str, Any]
    agent_sequence: List[AgentRoleEnum]

class WorkflowTemplateCreate(WorkflowTemplateBase):
    estimated_duration_hours: Optional[int] = Field(None, ge=1, le=168)

class WorkflowTemplateResponse(WorkflowTemplateBase, TimestampMixin):
    id: int
    is_active: bool
    is_default: bool
    estimated_duration_hours: Optional[int] = None
    times_used: int = 0
    success_rate: float = 0.0

class WorkflowExecutionRequest(BaseSchema):
    brief_id: int
    template_id: Optional[int] = None
    agent_sequence: Optional[List[AgentRoleEnum]] = None
    initial_prompt: str = Field(..., min_length=1)

class WorkflowExecutionResponse(BaseSchema):
    workflow_id: str
    brief_id: int
    status: str
    current_agent: Optional[AgentRoleEnum] = None
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    agent_logs: List[int] = Field(default_factory=list)  # Log IDs

# Search and Filter Schemas
class BriefSearchRequest(BaseSchema):
    query: Optional[str] = None
    case_type: Optional[str] = None
    status: Optional[BriefStatusEnum] = None
    jurisdiction: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    owner_id: Optional[int] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

class ResearchSearchRequest(BaseSchema):
    query: Optional[str] = None
    brief_id: Optional[int] = None
    source_type: Optional[SourceTypeEnum] = None
    min_relevance: Optional[int] = Field(None, ge=1, le=10)
    verified_only: bool = False
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

# Statistics and Analytics Schemas
class BriefStatsResponse(BaseSchema):
    total_briefs: int
    drafts: int
    in_progress: int
    completed: int
    avg_word_count: float
    avg_completion_days: float

class AgentStatsResponse(BaseSchema):
    agent_role: AgentRoleEnum
    total_executions: int
    success_rate: float
    avg_execution_time_ms: float
    total_tokens_used: int
    most_used_model: Optional[str] = None

class ModelStatsResponse(BaseSchema):
    model_name: str
    total_calls: int
    total_tokens: int
    avg_execution_time_ms: float
    success_rate: float
    preferred_agents: List[AgentRoleEnum]

# Error Response Schema
class ErrorResponse(BaseSchema):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Health Check Schema
class HealthCheckResponse(BaseSchema):
    status: str
    database: str
    local_llm: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"