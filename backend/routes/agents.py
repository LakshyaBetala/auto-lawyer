"""
AutoLawyer Agent Routes
FastAPI routes for agent execution, workflow management, and agent analytics
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import json

# Internal imports
from db.database import get_db
from db.models import User, Brief, AgentLog, ModelConfig, WorkflowTemplate, AgentRole, BriefStatus
from db import schemas
from core.agent_executor import AgentExecutor
from core.graph_builder import WorkflowGraph
from local_llm.llm_runner import LLMManager
from config import logger

# Import from main for dependency injection
from main import get_current_user, llm_manager, agent_executor, workflow_graph

router = APIRouter()

# Agent Execution Endpoints
@router.post("/execute", response_model=schemas.AgentExecutionResponse)
async def execute_agent(
    request: schemas.AgentExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Execute a single agent with the given prompt
    """
    try:
        # Validate brief exists and user has access
        brief = db.query(Brief).filter(
            and_(Brief.id == request.brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found or access denied"
            )
        
        # Check if agent executor is available
        if not agent_executor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent system not available"
            )
        
        # Determine which model to use
        model_name = request.model_override
        if not model_name:
            # Get preferred model for this agent role
            preferred_model = db.query(ModelConfig).filter(
                and_(
                    ModelConfig.is_active == True,
                    getattr(ModelConfig, f'preferred_for_{request.agent_role.value}') == True
                )
            ).first()
            
            if preferred_model:
                model_name = preferred_model.name
            else:
                # Fallback to default model
                default_model = db.query(ModelConfig).filter(
                    and_(ModelConfig.is_active == True, ModelConfig.is_default == True)
                ).first()
                if default_model:
                    model_name = default_model.name
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="No active models available"
                    )
        
        # Start execution timing
        start_time = datetime.now()
        
        # Execute the agent
        try:
            result = await agent_executor.execute_agent(
                agent_role=request.agent_role,
                prompt=request.prompt,
                brief_id=request.brief_id,
                user_id=current_user.id,
                model_name=model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Create agent log
            agent_log = AgentLog(
                brief_id=request.brief_id,
                user_id=current_user.id,
                agent_role=request.agent_role,
                agent_name=f"{request.agent_role.value}_agent",
                action="execute",
                input_prompt=request.prompt,
                output_content=result.get('output', ''),
                execution_time_ms=execution_time_ms,
                tokens_used=result.get('tokens_used', 0),
                model_used=model_name,
                success=result.get('success', False),
                error_message=result.get('error'),
                workflow_step="single_execution"
            )
            
            db.add(agent_log)
            db.commit()
            db.refresh(agent_log)
            
            # Update model usage statistics in background
            background_tasks.add_task(update_model_stats, model_name, execution_time_ms, result.get('tokens_used', 0))
            
            return schemas.AgentExecutionResponse(
                success=result.get('success', False),
                agent_role=request.agent_role,
                output=result.get('output'),
                execution_time_ms=execution_time_ms,
                tokens_used=result.get('tokens_used', 0),
                model_used=model_name,
                error_message=result.get('error'),
                log_id=agent_log.id
            )
            
        except Exception as e:
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Log the error
            agent_log = AgentLog(
                brief_id=request.brief_id,
                user_id=current_user.id,
                agent_role=request.agent_role,
                agent_name=f"{request.agent_role.value}_agent",
                action="execute",
                input_prompt=request.prompt,
                output_content="",
                execution_time_ms=execution_time_ms,
                tokens_used=0,
                model_used=model_name,
                success=False,
                error_message=str(e),
                workflow_step="single_execution"
            )
            
            db.add(agent_log)
            db.commit()
            db.refresh(agent_log)
            
            return schemas.AgentExecutionResponse(
                success=False,
                agent_role=request.agent_role,
                output=None,
                execution_time_ms=execution_time_ms,
                tokens_used=0,
                model_used=model_name,
                error_message=str(e),
                log_id=agent_log.id
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute agent"
        )

@router.post("/workflow/execute", response_model=schemas.WorkflowExecutionResponse)
async def execute_workflow(
    request: schemas.WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Execute a complete workflow with multiple agents
    """
    try:
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == request.brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found or access denied"
            )
        
        if not workflow_graph:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Workflow system not available"
            )
        
        # Get workflow template if specified
        workflow_template = None
        agent_sequence = request.agent_sequence or []
        
        if request.template_id:
            workflow_template = db.query(WorkflowTemplate).filter(
                WorkflowTemplate.id == request.template_id
            ).first()
            
            if not workflow_template:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow template not found"
                )
            
            agent_sequence = workflow_template.agent_sequence
        
        # Default sequence if none provided
        if not agent_sequence:
            agent_sequence = [AgentRole.RESEARCHER, AgentRole.DRAFTER, AgentRole.SUMMARIZER]
        
        # Update brief status
        brief.status = BriefStatus.IN_PROGRESS
        db.commit()
        
        # Generate workflow ID
        workflow_id = f"wf_{brief.id}_{int(datetime.now().timestamp())}"
        
        # Execute workflow in background
        background_tasks.add_task(
            execute_workflow_background,
            workflow_id=workflow_id,
            brief_id=request.brief_id,
            user_id=current_user.id,
            agent_sequence=agent_sequence,
            initial_prompt=request.initial_prompt,
            template_id=request.template_id
        )
        
        # Estimate completion time
        estimated_completion = datetime.now() + timedelta(minutes=len(agent_sequence) * 5)
        
        return schemas.WorkflowExecutionResponse(
            workflow_id=workflow_id,
            brief_id=request.brief_id,
            status="started",
            current_agent=agent_sequence[0] if agent_sequence else None,
            progress_percentage=0.0,
            started_at=datetime.now(),
            estimated_completion=estimated_completion,
            agent_logs=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow execution error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start workflow execution"
        )

@router.get("/workflow/{workflow_id}/status", response_model=schemas.WorkflowExecutionResponse)
async def get_workflow_status(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the status of a running workflow
    """
    try:
        # Extract brief_id from workflow_id
        parts = workflow_id.split('_')
        if len(parts) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid workflow ID format"
            )
        
        brief_id = int(parts[1])
        
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found or access denied"
            )
        
        # Get agent logs for this workflow
        workflow_logs = db.query(AgentLog).filter(
            and_(
                AgentLog.brief_id == brief_id,
                AgentLog.workflow_step.like(f"%{workflow_id}%")
            )
        ).order_by(AgentLog.created_at).all()
        
        if not workflow_logs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow not found"
            )
        
        # Calculate progress
        total_steps = len(set(log.agent_role for log in workflow_logs))
        completed_steps = len([log for log in workflow_logs if log.success and log.completed_at])
        progress_percentage = (completed_steps / max(total_steps, 1)) * 100
        
        # Determine current status
        failed_logs = [log for log in workflow_logs if not log.success]
        if failed_logs:
            status_str = "failed"
            current_agent = None
        elif progress_percentage >= 100:
            status_str = "completed"
            current_agent = None
        else:
            status_str = "running"
            # Find current agent (last incomplete step)
            incomplete_logs = [log for log in workflow_logs if not log.completed_at]
            current_agent = incomplete_logs[-1].agent_role if incomplete_logs else None
        
        return schemas.WorkflowExecutionResponse(
            workflow_id=workflow_id,
            brief_id=brief_id,
            status=status_str,
            current_agent=current_agent,
            progress_percentage=progress_percentage,
            started_at=workflow_logs[0].created_at if workflow_logs else datetime.now(),
            estimated_completion=None,  # Could calculate based on remaining steps
            agent_logs=[log.id for log in workflow_logs]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow status"
        )

# Agent Log Endpoints
@router.get("/logs", response_model=List[schemas.AgentLogResponse])
async def get_agent_logs(
    brief_id: Optional[int] = Query(None, description="Filter by brief ID"),
    agent_role: Optional[schemas.AgentRoleEnum] = Query(None, description="Filter by agent role"),
    success_only: bool = Query(False, description="Show only successful executions"),
    limit: int = Query(50, ge=1, le=500, description="Number of logs to return"),
    offset: int = Query(0, ge=0, description="Number of logs to skip"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get agent execution logs with filtering
    """
    try:
        # Build query
        query = db.query(AgentLog).filter(AgentLog.user_id == current_user.id)
        
        if brief_id:
            # Validate brief access
            brief = db.query(Brief).filter(
                and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
            ).first()
            if not brief:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Brief not found or access denied"
                )
            query = query.filter(AgentLog.brief_id == brief_id)
        
        if agent_role:
            query = query.filter(AgentLog.agent_role == agent_role)
        
        if success_only:
            query = query.filter(AgentLog.success == True)
        
        # Order by most recent first
        logs = query.order_by(desc(AgentLog.created_at)).offset(offset).limit(limit).all()
        
        return logs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get agent logs error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent logs"
        )

@router.get("/logs/{log_id}", response_model=schemas.AgentLogResponse)
async def get_agent_log(
    log_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific agent log by ID
    """
    try:
        log = db.query(AgentLog).filter(
            and_(AgentLog.id == log_id, AgentLog.user_id == current_user.id)
        ).first()
        
        if not log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent log not found"
            )
        
        return log
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get agent log error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent log"
        )

# Agent Analytics Endpoints
@router.get("/stats", response_model=List[schemas.AgentStatsResponse])
async def get_agent_statistics(
    days_back: int = Query(30, ge=1, le=365, description="Number of days to include in stats"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get agent performance statistics
    """
    try:
        since_date = datetime.now() - timedelta(days=days_back)
        
        # Query agent stats grouped by role
        stats_query = db.query(
            AgentLog.agent_role,
            func.count(AgentLog.id).label('total_executions'),
            func.avg(func.cast(AgentLog.success, db.Integer)).label('success_rate'),
            func.avg(AgentLog.execution_time_ms).label('avg_execution_time'),
            func.sum(AgentLog.tokens_used).label('total_tokens'),
            func.mode().within_group(AgentLog.model_used).label('most_used_model')
        ).filter(
            and_(
                AgentLog.user_id == current_user.id,
                AgentLog.created_at >= since_date
            )
        ).group_by(AgentLog.agent_role).all()
        
        # Convert to response format
        agent_stats = []
        for stat in stats_query:
            agent_stats.append(schemas.AgentStatsResponse(
                agent_role=stat.agent_role,
                total_executions=stat.total_executions,
                success_rate=float(stat.success_rate or 0),
                avg_execution_time_ms=float(stat.avg_execution_time or 0),
                total_tokens_used=int(stat.total_tokens or 0),
                most_used_model=stat.most_used_model
            ))
        
        return agent_stats
        
    except Exception as e:
        logger.error(f"Agent statistics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent statistics"
        )

@router.get("/models", response_model=List[schemas.ModelConfigResponse])
async def get_available_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get available LLM models and their configurations
    """
    try:
        models = db.query(ModelConfig).filter(ModelConfig.is_active == True).all()
        return models
        
    except Exception as e:
        logger.error(f"Get models error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve available models"
        )

# Background task functions
async def execute_workflow_background(
    workflow_id: str,
    brief_id: int,
    user_id: int,
    agent_sequence: List[AgentRole],
    initial_prompt: str,
    template_id: Optional[int] = None
):
    """
    Execute workflow in background
    """
    try:
        if not workflow_graph:
            logger.error("Workflow graph not available")
            return
        
        # Execute the workflow
        await workflow_graph.execute_workflow(
            workflow_id=workflow_id,
            brief_id=brief_id,
            user_id=user_id,
            agent_sequence=agent_sequence,
            initial_prompt=initial_prompt,
            template_id=template_id
        )
        
        logger.info(f"Workflow {workflow_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Background workflow execution failed: {str(e)}")

async def update_model_stats(model_name: str, execution_time_ms: int, tokens_used: int):
    """
    Update model usage statistics
    """
    try:
        with SessionLocal() as db:
            model = db.query(ModelConfig).filter(ModelConfig.name == model_name).first()
            if model:
                model.total_calls += 1
                model.total_tokens += tokens_used
                db.commit()
        
    except Exception as e:
        logger.error(f"Failed to update model stats: {str(e)}")

# Import SessionLocal for background tasks
from db.database import SessionLocal