import logging
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
import uuid

from backend.config import get_config
from backend.agents.researcher import LegalResearcher, ResearchQuery, ResearchResult, create_researcher
from backend.agents.drafter import LegalDrafter, DraftingRequest, DraftResult, create_drafter
from backend.agents.summarizer import LegalSummarizer, SummaryRequest, SummaryResult, create_summarizer

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class AgentStatus(Enum):
    """Individual agent status"""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

class WorkflowType(Enum):
    """Predefined workflow types"""
    RESEARCH_ONLY = "research_only"
    DRAFT_ONLY = "draft_only"
    SUMMARIZE_ONLY = "summarize_only"
    RESEARCH_AND_DRAFT = "research_and_draft"
    RESEARCH_AND_SUMMARIZE = "research_and_summarize"
    DRAFT_AND_SUMMARIZE = "draft_and_summarize"
    FULL_PIPELINE = "full_pipeline"
    CUSTOM = "custom"

@dataclass
class AgentInfo:
    """Information about a registered agent"""
    agent_id: str
    agent_name: str
    agent_type: str
    instance: Any
    status: AgentStatus = AgentStatus.AVAILABLE
    last_used: datetime = field(default_factory=datetime.now)
    total_executions: int = 0
    average_execution_time: float = 0.0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "last_used": self.last_used.isoformat(),
            "total_executions": self.total_executions,
            "average_execution_time": self.average_execution_time,
            "error_count": self.error_count
        }

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    step_id: str
    agent_id: str
    action: str
    inputs: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 2
    
    # Execution metadata
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "retry_count": self.retry_count
        }

@dataclass
class Workflow:
    """Complete workflow definition and execution state"""
    workflow_id: str
    workflow_type: WorkflowType
    name: str
    description: str
    steps: List[WorkflowStep]
    
    # Execution metadata
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_step: Optional[str] = None
    progress_percentage: float = 0.0
    
    # Results aggregation
    final_results: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type.value,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "current_step": self.current_step,
            "progress_percentage": self.progress_percentage,
            "steps": [step.to_dict() for step in self.steps],
            "final_results": self.final_results,
            "step_count": len(self.steps),
            "completed_steps": len([s for s in self.steps if s.status == WorkflowStatus.COMPLETED])
        }

class AgentRegistry:
    """
    Central registry and orchestrator for all AutoLawyer agents
    Manages agent lifecycle, workflow execution, and inter-agent communication
    """
    
    def __init__(self):
        """Initialize the Agent Registry"""
        self.config = get_config()
        self.agents: Dict[str, AgentInfo] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.llm.max_concurrent_requests)
        
        # Performance tracking
        self.total_workflows_executed = 0
        self.total_execution_time = 0.0
        
        # Initialize and register default agents
        self._initialize_default_agents()
        
        logger.info("Agent Registry initialized successfully")
    
    def _initialize_default_agents(self):
        """Initialize and register the core agents"""
        try:
            # Register Researcher
            researcher = create_researcher()
            self.register_agent(
                agent_id="researcher",
                agent_name="Legal Researcher",
                agent_type="researcher",
                instance=researcher
            )
            
            # Register Drafter
            drafter = create_drafter()
            self.register_agent(
                agent_id="drafter",
                agent_name="Legal Drafter", 
                agent_type="drafter",
                instance=drafter
            )
            
            # Register Summarizer
            summarizer = create_summarizer()
            self.register_agent(
                agent_id="summarizer",
                agent_name="Legal Summarizer",
                agent_type="summarizer", 
                instance=summarizer
            )
            
            logger.info("All default agents registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize default agents: {e}")
            raise
    
    def register_agent(self, 
                      agent_id: str, 
                      agent_name: str, 
                      agent_type: str, 
                      instance: Any) -> bool:
        """
        Register a new agent in the registry
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name
            agent_type: Type/category of agent
            instance: Agent instance object
            
        Returns:
            bool: True if registration successful
        """
        
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered, updating...")
        
        agent_info = AgentInfo(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            instance=instance
        )
        
        self.agents[agent_id] = agent_info
        logger.info(f"Agent registered: {agent_id} ({agent_name})")
        
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent unregistered: {agent_id}")
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information by ID"""
        return self.agents.get(agent_id)
    
    def get_available_agents(self) -> List[AgentInfo]:
        """Get list of all available agents"""
        return [agent for agent in self.agents.values() 
                if agent.status == AgentStatus.AVAILABLE]
    
    def _update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].last_used = datetime.now()
    
    def _execute_agent_action(self, 
                             agent_id: str, 
                             action: str, 
                             inputs: Dict[str, Any]) -> Any:
        """
        Execute a specific action on an agent
        
        Args:
            agent_id: ID of agent to execute
            action: Action method name to call
            inputs: Input parameters for the action
            
        Returns:
            Result of the agent action
        """
        
        agent_info = self.get_agent(agent_id)
        if not agent_info:
            raise ValueError(f"Agent {agent_id} not found")
        
        if agent_info.status != AgentStatus.AVAILABLE:
            raise RuntimeError(f"Agent {agent_id} is not available (status: {agent_info.status.value})")
        
        # Update agent status
        self._update_agent_status(agent_id, AgentStatus.BUSY)
        
        try:
            # Get the method from the agent instance
            method = getattr(agent_info.instance, action)
            if not callable(method):
                raise AttributeError(f"Action {action} is not callable on agent {agent_id}")
            
            # Execute the action
            start_time = time.time()
            result = method(**inputs)
            execution_time = time.time() - start_time
            
            # Update agent statistics
            agent_info.total_executions += 1
            if agent_info.total_executions == 1:
                agent_info.average_execution_time = execution_time
            else:
                # Update running average
                agent_info.average_execution_time = (
                    (agent_info.average_execution_time * (agent_info.total_executions - 1) + execution_time) 
                    / agent_info.total_executions
                )
            
            logger.info(f"Agent {agent_id} executed {action} in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            agent_info.error_count += 1
            logger.error(f"Agent {agent_id} action {action} failed: {e}")
            raise
        
        finally:
            # Always reset agent status
            self._update_agent_status(agent_id, AgentStatus.AVAILABLE)
    
    def create_workflow(self, 
                       workflow_type: WorkflowType,
                       name: str,
                       description: str = "",
                       custom_steps: List[WorkflowStep] = None) -> str:
        """
        Create a new workflow
        
        Args:
            workflow_type: Type of workflow to create
            name: Workflow name
            description: Workflow description
            custom_steps: Custom steps for CUSTOM workflow type
            
        Returns:
            str: Workflow ID
        """
        
        workflow_id = f"workflow_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Generate steps based on workflow type
        if workflow_type == WorkflowType.CUSTOM and custom_steps:
            steps = custom_steps
        else:
            steps = self._generate_workflow_steps(workflow_type)
        
        workflow = Workflow(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            name=name,
            description=description,
            steps=steps
        )
        
        self.workflows[workflow_id] = workflow
        
        logger.info(f"Workflow created: {workflow_id} ({workflow_type.value})")
        return workflow_id
    
    def _generate_workflow_steps(self, workflow_type: WorkflowType) -> List[WorkflowStep]:
        """Generate workflow steps based on type"""
        
        steps = []
        
        if workflow_type == WorkflowType.RESEARCH_ONLY:
            steps.append(WorkflowStep(
                step_id="research_1",
                agent_id="researcher",
                action="research",
                inputs={}
            ))
        
        elif workflow_type == WorkflowType.DRAFT_ONLY:
            steps.append(WorkflowStep(
                step_id="draft_1",
                agent_id="drafter",
                action="draft_document",
                inputs={}
            ))
        
        elif workflow_type == WorkflowType.SUMMARIZE_ONLY:
            steps.append(WorkflowStep(
                step_id="summarize_1",
                agent_id="summarizer",
                action="summarize",
                inputs={}
            ))
        
        elif workflow_type == WorkflowType.RESEARCH_AND_DRAFT:
            steps.extend([
                WorkflowStep(
                    step_id="research_1",
                    agent_id="researcher",
                    action="research",
                    inputs={}
                ),
                WorkflowStep(
                    step_id="draft_1",
                    agent_id="drafter",
                    action="draft_document",
                    inputs={},
                    dependencies=["research_1"]
                )
            ])
        
        elif workflow_type == WorkflowType.RESEARCH_AND_SUMMARIZE:
            steps.extend([
                WorkflowStep(
                    step_id="research_1",
                    agent_id="researcher",
                    action="research",
                    inputs={}
                ),
                WorkflowStep(
                    step_id="summarize_1",
                    agent_id="summarizer",
                    action="summarize",
                    inputs={},
                    dependencies=["research_1"]
                )
            ])
        
        elif workflow_type == WorkflowType.DRAFT_AND_SUMMARIZE:
            steps.extend([
                WorkflowStep(
                    step_id="draft_1",
                    agent_id="drafter",
                    action="draft_document",
                    inputs={}
                ),
                WorkflowStep(
                    step_id="summarize_1",
                    agent_id="summarizer",
                    action="summarize",
                    inputs={},
                    dependencies=["draft_1"]
                )
            ])
        
        elif workflow_type == WorkflowType.FULL_PIPELINE:
            steps.extend([
                WorkflowStep(
                    step_id="research_1",
                    agent_id="researcher",
                    action="research",
                    inputs={}
                ),
                WorkflowStep(
                    step_id="draft_1",
                    agent_id="drafter",
                    action="draft_document",
                    inputs={},
                    dependencies=["research_1"]
                ),
                WorkflowStep(
                    step_id="summarize_1",
                    agent_id="summarizer",
                    action="summarize",
                    inputs={},
                    dependencies=["draft_1"]
                )
            ])
        
        return steps
    
    def execute_workflow(self, 
                        workflow_id: str, 
                        initial_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a workflow synchronously
        
        Args:
            workflow_id: ID of workflow to execute
            initial_inputs: Initial inputs for the workflow
            
        Returns:
            Dict containing workflow results
        """
        
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        logger.info(f"Starting workflow execution: {workflow_id}")
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.start_time = datetime.now()
        
        try:
            # Execute steps in dependency order
            completed_steps = {}
            
            for step in workflow.steps:
                # Check if dependencies are satisfied
                if not self._dependencies_satisfied(step, completed_steps):
                    continue
                
                # Execute step
                logger.info(f"Executing step: {step.step_id}")
                workflow.current_step = step.step_id
                
                step.status = WorkflowStatus.RUNNING
                step.start_time = datetime.now()
                
                try:
                    # Prepare step inputs
                    step_inputs = step.inputs.copy()
                    
                    # Add dependency results
                    for dep_step_id in step.dependencies:
                        if dep_step_id in completed_steps:
                            step_inputs[f"{dep_step_id}_result"] = completed_steps[dep_step_id]
                    
                    # Add initial inputs if this is the first step
                    if not step.dependencies and initial_inputs:
                        step_inputs.update(initial_inputs)
                    
                    # Execute the step
                    result = self._execute_agent_action(
                        step.agent_id,
                        step.action,
                        step_inputs
                    )
                    
                    step.result = result
                    step.status = WorkflowStatus.COMPLETED
                    step.end_time = datetime.now()
                    
                    completed_steps[step.step_id] = result
                    workflow.intermediate_results[step.step_id] = result
                    
                    # Update progress
                    workflow.progress_percentage = (len(completed_steps) / len(workflow.steps)) * 100
                    
                    logger.info(f"Step {step.step_id} completed successfully")
                    
                except Exception as e:
                    step.error = str(e)
                    step.status = WorkflowStatus.FAILED
                    step.end_time = datetime.now()
                    
                    # Retry logic
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        logger.warning(f"Step {step.step_id} failed, retrying ({step.retry_count}/{step.max_retries})")
                        # Reset step for retry
                        step.status = WorkflowStatus.PENDING
                        step.start_time = None
                        step.end_time = None
                        continue
                    else:
                        logger.error(f"Step {step.step_id} failed permanently: {e}")
                        workflow.status = WorkflowStatus.FAILED
                        raise
            
            # Workflow completed successfully
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.now()
            workflow.progress_percentage = 100.0
            
            # Aggregate final results
            workflow.final_results = {
                "workflow_id": workflow_id,
                "execution_time_seconds": (workflow.end_time - workflow.start_time).total_seconds(),
                "steps_completed": len(completed_steps),
                "results": completed_steps
            }
            
            self.total_workflows_executed += 1
            self.total_execution_time += workflow.final_results["execution_time_seconds"]
            
            logger.info(f"Workflow {workflow_id} completed successfully in {workflow.final_results['execution_time_seconds']:.2f}s")
            
            return workflow.final_results
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
            logger.error(f"Workflow {workflow_id} failed: {e}")
            raise
    
    def _dependencies_satisfied(self, step: WorkflowStep, completed_steps: Dict[str, Any]) -> bool:
        """Check if step dependencies are satisfied"""
        return all(dep in completed_steps for dep in step.dependencies)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status and progress"""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            return workflow.to_dict()
        return None
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        workflow = self.workflows.get(workflow_id)
        if workflow and workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.CANCELLED
            workflow.end_time = datetime.now()
            logger.info(f"Workflow {workflow_id} cancelled")
            return True
        return False
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status and statistics"""
        return {
            "total_agents": len(self.agents),
            "available_agents": len(self.get_available_agents()),
            "total_workflows": len(self.workflows),
            "active_workflows": len([w for w in self.workflows.values() 
                                   if w.status == WorkflowStatus.RUNNING]),
            "completed_workflows": len([w for w in self.workflows.values() 
                                      if w.status == WorkflowStatus.COMPLETED]),
            "failed_workflows": len([w for w in self.workflows.values() 
                                   if w.status == WorkflowStatus.FAILED]),
            "total_workflows_executed": self.total_workflows_executed,
            "average_workflow_time": (self.total_execution_time / max(self.total_workflows_executed, 1)),
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "supported_workflow_types": [wt.value for wt in WorkflowType]
        }
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24) -> int:
        """Clean up old completed workflows"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        workflows_to_remove = [
            wf_id for wf_id, workflow in self.workflows.items()
            if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
            and workflow.end_time and workflow.end_time < cutoff_time
        ]
        
        for wf_id in workflows_to_remove:
            del self.workflows[wf_id]
        
        logger.info(f"Cleaned up {len(workflows_to_remove)} old workflows")
        return len(workflows_to_remove)

# Global registry instance
_registry: Optional[AgentRegistry] = None

def get_registry() -> AgentRegistry:
    """Get or create the global agent registry instance"""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry

def create_quick_workflow(workflow_type: str, **kwargs) -> str:
    """Quick workflow creation helper"""
    registry = get_registry()
    
    wf_type = WorkflowType(workflow_type)
    name = kwargs.get('name', f'{workflow_type.replace("_", " ").title()} Workflow')
    description = kwargs.get('description', f'Auto-generated {workflow_type} workflow')
    
    return registry.create_workflow(wf_type, name, description)

# Convenience functions for common workflows
def research_and_draft_workflow(research_query: str, draft_specs: Dict[str, Any]) -> str:
    """Create and execute a research + draft workflow"""
    registry = get_registry()
    
    workflow_id = create_quick_workflow("research_and_draft")
    
    # Execute with specific inputs
    initial_inputs = {
        "query": research_query,
        **draft_specs
    }
    
    return registry.execute_workflow(workflow_id, initial_inputs)

def full_legal_analysis_workflow(legal_question: str) -> str:
    """Create and execute a complete legal analysis workflow"""
    registry = get_registry()
    
    workflow_id = create_quick_workflow("full_pipeline", 
                                       name="Complete Legal Analysis",
                                       description="Research, draft, and summarize legal analysis")
    
    initial_inputs = {"query": legal_question}
    
    return registry.execute_workflow(workflow_id, initial_inputs)