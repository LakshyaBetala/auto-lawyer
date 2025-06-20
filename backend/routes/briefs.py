"""
AutoLawyer Brief Routes
FastAPI routes for legal brief management, version control, and research integration
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks, File, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import re
from pathlib import Path

# Internal imports
from db.database import get_db
from db.models import (
    User, Brief, BriefVersion, AgentLog, ResearchResult, 
    BriefStatus, AgentRole, SourceTypeEnum
)
from db import schemas
from config import logger

# Import from main for dependency injection
from main import get_current_user

router = APIRouter()

# Brief CRUD Endpoints
@router.post("/", response_model=schemas.BriefResponse)
async def create_brief(
    brief_data: schemas.BriefCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new legal brief
    """
    try:
        # Create new brief
        new_brief = Brief(
            title=brief_data.title,
            description=brief_data.description,
            case_type=brief_data.case_type,
            jurisdiction=brief_data.jurisdiction,
            court_level=brief_data.court_level,
            content=brief_data.content or "",
            research_notes=brief_data.research_notes or "",
            citations=brief_data.citations or [],
            status=BriefStatus.DRAFT,
            owner_id=current_user.id,
            word_count=len(brief_data.content.split()) if brief_data.content else 0,
            page_count=max(1, len(brief_data.content.split()) // 250) if brief_data.content else 1
        )
        
        db.add(new_brief)
        db.commit()
        db.refresh(new_brief)
        
        # Create initial version
        if brief_data.content:
            initial_version = BriefVersion(
                brief_id=new_brief.id,
                version_number=1,
                title=brief_data.title,
                content=brief_data.content,
                changes_summary="Initial brief creation",
                word_count=new_brief.word_count
            )
            db.add(initial_version)
            db.commit()
        
        logger.info(f"Created new brief {new_brief.id} for user {current_user.id}")
        
        return new_brief
        
    except Exception as e:
        logger.error(f"Brief creation error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create brief"
        )

@router.get("/", response_model=List[schemas.BriefSummary])
async def get_briefs(
    skip: int = Query(0, ge=0, description="Number of briefs to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of briefs to return"),
    status_filter: Optional[schemas.BriefStatusEnum] = Query(None, description="Filter by status"),
    case_type: Optional[str] = Query(None, description="Filter by case type"),
    search: Optional[str] = Query(None, description="Search in title and description"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's briefs with filtering and pagination
    """
    try:
        # Build query
        query = db.query(Brief).filter(Brief.owner_id == current_user.id)
        
        # Apply filters
        if status_filter:
            query = query.filter(Brief.status == status_filter)
        
        if case_type:
            query = query.filter(Brief.case_type.ilike(f"%{case_type}%"))
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Brief.title.ilike(search_term),
                    Brief.description.ilike(search_term),
                    Brief.content.ilike(search_term)
                )
            )
        
        # Order by most recent first
        briefs = query.order_by(desc(Brief.updated_at)).offset(skip).limit(limit).all()
        
        # Convert to summary format
        brief_summaries = []
        for brief in briefs:
            brief_summaries.append(schemas.BriefSummary(
                id=brief.id,
                title=brief.title,
                case_type=brief.case_type,
                status=brief.status,
                word_count=brief.word_count,
                created_at=brief.created_at,
                updated_at=brief.updated_at
            ))
        
        return brief_summaries
        
    except Exception as e:
        logger.error(f"Get briefs error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve briefs"
        )

@router.get("/{brief_id}", response_model=schemas.BriefResponse)
async def get_brief(
    brief_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific brief by ID
    """
    try:
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        return brief
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get brief error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve brief"
        )

@router.put("/{brief_id}", response_model=schemas.BriefResponse)
async def update_brief(
    brief_id: int,
    brief_update: schemas.BriefUpdate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing brief
    """
    try:
        # Get existing brief
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        # Store original content for version tracking
        original_content = brief.content
        content_changed = False
        
        # Update fields
        update_data = brief_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(brief, field):
                setattr(brief, field, value)
                if field == "content" and value != original_content:
                    content_changed = True
        
        # Update word and page counts if content changed
        if content_changed and brief.content:
            brief.word_count = len(brief.content.split())
            brief.page_count = max(1, brief.word_count // 250)
        
        brief.updated_at = datetime.now()
        
        db.commit()
        db.refresh(brief)
        
        # Create new version if content changed significantly
        if content_changed and brief.content:
            background_tasks.add_task(
                create_brief_version,
                brief_id=brief.id,
                content=brief.content,
                title=brief.title,
                changes_summary="Manual edit"
            )
        
        logger.info(f"Updated brief {brief_id} for user {current_user.id}")
        
        return brief
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update brief error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update brief"
        )

@router.delete("/{brief_id}")
async def delete_brief(
    brief_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a brief (soft delete - mark as archived)
    """
    try:
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        # Soft delete by marking as archived
        brief.status = BriefStatus.ARCHIVED
        brief.updated_at = datetime.now()
        
        db.commit()
        
        logger.info(f"Archived brief {brief_id} for user {current_user.id}")
        
        return {"message": "Brief archived successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete brief error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to archive brief"
        )

# Brief Version Management
@router.get("/{brief_id}/versions", response_model=List[schemas.BriefVersionResponse])
async def get_brief_versions(
    brief_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all versions of a brief
    """
    try:
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        versions = db.query(BriefVersion).filter(
            BriefVersion.brief_id == brief_id
        ).order_by(desc(BriefVersion.version_number)).all()
        
        return versions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get brief versions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve brief versions"
        )

@router.post("/{brief_id}/versions", response_model=schemas.BriefVersionResponse)
async def create_version(
    brief_id: int,
    version_data: schemas.BriefVersionBase,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new version of a brief
    """
    try:
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        # Get next version number
        latest_version = db.query(func.max(BriefVersion.version_number)).filter(
            BriefVersion.brief_id == brief_id
        ).scalar() or 0
        
        # Create new version
        new_version = BriefVersion(
            brief_id=brief_id,
            version_number=latest_version + 1,
            title=version_data.title or brief.title,
            content=version_data.content,
            changes_summary=version_data.changes_summary,
            created_by_agent=version_data.created_by_agent,
            word_count=len(version_data.content.split())
        )
        
        db.add(new_version)
        
        # Update main brief with latest content
        brief.content = version_data.content
        brief.word_count = new_version.word_count
        brief.page_count = max(1, new_version.word_count // 250)
        brief.updated_at = datetime.now()
        
        db.commit()
        db.refresh(new_version)
        
        logger.info(f"Created version {new_version.version_number} for brief {brief_id}")
        
        return new_version
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create version error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create brief version"
        )

@router.post("/{brief_id}/restore/{version_number}")
async def restore_version(
    brief_id: int,
    version_number: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Restore a brief to a specific version
    """
    try:
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        # Get the version to restore
        version = db.query(BriefVersion).filter(
            and_(
                BriefVersion.brief_id == brief_id,
                BriefVersion.version_number == version_number
            )
        ).first()
        
        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Version not found"
            )
        
        # Create new version with restored content
        latest_version = db.query(func.max(BriefVersion.version_number)).filter(
            BriefVersion.brief_id == brief_id
        ).scalar() or 0
        
        restored_version = BriefVersion(
            brief_id=brief_id,
            version_number=latest_version + 1,
            title=version.title,
            content=version.content,
            changes_summary=f"Restored from version {version_number}",
            word_count=version.word_count
        )
        
        db.add(restored_version)
        
        # Update main brief
        brief.content = version.content
        brief.title = version.title or brief.title
        brief.word_count = version.word_count
        brief.page_count = max(1, version.word_count // 250)
        brief.updated_at = datetime.now()
        
        db.commit()
        
        logger.info(f"Restored brief {brief_id} to version {version_number}")
        
        return {"message": f"Brief restored to version {version_number}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Restore version error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to restore brief version"
        )

# Research Integration
@router.get("/{brief_id}/research", response_model=List[schemas.ResearchResultResponse])
async def get_research_results(
    brief_id: int,
    source_type: Optional[schemas.SourceTypeEnum] = Query(None, description="Filter by source type"),
    min_relevance: Optional[int] = Query(None, ge=1, le=10, description="Minimum relevance score"),
    verified_only: bool = Query(False, description="Show only verified results"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get research results for a brief
    """
    try:
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        # Build query
        query = db.query(ResearchResult).filter(ResearchResult.brief_id == brief_id)
        
        if source_type:
            query = query.filter(ResearchResult.source_type == source_type)
        
        if min_relevance:
            query = query.filter(ResearchResult.relevance_score >= min_relevance)
        
        if verified_only:
            query = query.filter(ResearchResult.verified == True)
        
        # Order by relevance score (highest first)
        results = query.order_by(desc(ResearchResult.relevance_score)).all()
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get research results error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve research results"
        )

@router.post("/{brief_id}/research", response_model=schemas.ResearchResultResponse)
async def add_research_result(
    brief_id: int,
    research_data: schemas.ResearchResultCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add a research result to a brief
    """
    try:
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        # Create research result
        research_result = ResearchResult(
            brief_id=brief_id,
            query=research_data.query,
            source_type=research_data.source_type,
            title=research_data.title,
            citation=research_data.citation,
            court=research_data.court,
            year=research_data.year,
            summary=research_data.summary,
            relevant_excerpt=research_data.relevant_excerpt,
            full_text=research_data.full_text,
            relevance_score=research_data.relevance_score,
            agent_notes=research_data.agent_notes,
            found_by_agent=research_data.found_by_agent
        )
        
        db.add(research_result)
        db.commit()
        db.refresh(research_result)
        
        logger.info(f"Added research result {research_result.id} to brief {brief_id}")
        
        return research_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add research result error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add research result"
        )

@router.put("/{brief_id}/research/{research_id}", response_model=schemas.ResearchResultResponse)
async def update_research_result(
    brief_id: int,
    research_id: int,
    research_update: schemas.ResearchResultUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a research result
    """
    try:
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        # Get research result
        research_result = db.query(ResearchResult).filter(
            and_(
                ResearchResult.id == research_id,
                ResearchResult.brief_id == brief_id
            )
        ).first()
        
        if not research_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Research result not found"
            )
        
        # Update fields
        update_data = research_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(research_result, field):
                setattr(research_result, field, value)
        
        db.commit()
        db.refresh(research_result)
        
        logger.info(f"Updated research result {research_id} for brief {brief_id}")
        
        return research_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update research result error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update research result"
        )

# Brief Analytics and Statistics
@router.get("/{brief_id}/analytics")
async def get_brief_analytics(
    brief_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get analytics for a specific brief
    """
    try:
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        # Agent execution stats
        agent_stats = db.query(
            AgentLog.agent_role,
            func.count(AgentLog.id).label('executions'),
            func.avg(AgentLog.execution_time_ms).label('avg_time'),
            func.sum(AgentLog.tokens_used).label('total_tokens')
        ).filter(AgentLog.brief_id == brief_id).group_by(AgentLog.agent_role).all()
        
        # Version stats
        version_count = db.query(BriefVersion).filter(BriefVersion.brief_id == brief_id).count()
        
        # Research stats
        research_stats = db.query(
            func.count(ResearchResult.id).label('total_results'),
            func.avg(ResearchResult.relevance_score).label('avg_relevance'),
            func.count(func.distinct(ResearchResult.source_type)).label('source_types')
        ).filter(ResearchResult.brief_id == brief_id).first()
        
        # Timeline data
        timeline = db.query(
            AgentLog.created_at,
            AgentLog.agent_role,
            AgentLog.action
        ).filter(AgentLog.brief_id == brief_id).order_by(AgentLog.created_at).all()
        
        return {
            "brief_info": {
                "id": brief.id,
                "title": brief.title,
                "status": brief.status,
                "word_count": brief.word_count,
                "created_at": brief.created_at,
                "updated_at": brief.updated_at
            },
            "agent_statistics": [
                {
                    "agent_role": stat.agent_role,
                    "executions": stat.executions,
                    "avg_execution_time_ms": float(stat.avg_time or 0),
                    "total_tokens": int(stat.total_tokens or 0)
                } for stat in agent_stats
            ],
            "version_count": version_count,
            "research_statistics": {
                "total_results": research_stats.total_results or 0,
                "average_relevance": float(research_stats.avg_relevance or 0),
                "source_types_count": research_stats.source_types or 0
            },
            "timeline": [
                {
                    "timestamp": event.created_at,
                    "agent_role": event.agent_role,
                    "action": event.action
                } for event in timeline
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get brief analytics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve brief analytics"
        )

@router.get("/statistics/overview", response_model=schemas.BriefStatsResponse)
async def get_brief_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get overview statistics for user's briefs
    """
    try:
        # Count briefs by status
        total_briefs = db.query(Brief).filter(Brief.owner_id == current_user.id).count()
        
        status_counts = db.query(
            Brief.status,
            func.count(Brief.id).label('count')
        ).filter(Brief.owner_id == current_user.id).group_by(Brief.status).all()
        
        # Calculate averages
        avg_stats = db.query(
            func.avg(Brief.word_count).label('avg_word_count'),
            func.avg(
                func.julianday(Brief.completed_at) - func.julianday(Brief.created_at)
            ).label('avg_completion_days')
        ).filter(
            and_(
                Brief.owner_id == current_user.id,
                Brief.completed_at.isnot(None)
            )
        ).first()
        
        # Convert status counts to dict
        status_dict = {status.value: 0 for status in BriefStatus}
        for status_count in status_counts:
            status_dict[status_count.status.value] = status_count.count
        
        return schemas.BriefStatsResponse(
            total_briefs=total_briefs,
            drafts=status_dict.get('draft', 0),
            in_progress=status_dict.get('in_progress', 0),
            completed=status_dict.get('completed', 0),
            avg_word_count=float(avg_stats.avg_word_count or 0),
            avg_completion_days=float(avg_stats.avg_completion_days or 0)
        )
        
    except Exception as e:
        logger.error(f"Get brief statistics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve brief statistics"
        )

# Background task functions
async def create_brief_version(brief_id: int, content: str, title: str, changes_summary: str):
    """
    Create a brief version in background
    """
    try:
        from db.database import SessionLocal
        
        with SessionLocal() as db:
            # Get next version number
            latest_version = db.query(func.max(BriefVersion.version_number)).filter(
                BriefVersion.brief_id == brief_id
            ).scalar() or 0
            
            # Create new version
            new_version = BriefVersion(
                brief_id=brief_id,
                version_number=latest_version + 1,
                title=title,
                content=content,
                changes_summary=changes_summary,
                word_count=len(content.split())
            )
            
            db.add(new_version)
            db.commit()
            
            logger.info(f"Background: Created version {new_version.version_number} for brief {brief_id}")
            
    except Exception as e:
        logger.error(f"Background version creation failed: {str(e)}")

# Export endpoints
@router.get("/{brief_id}/export/{format}")
async def export_brief(
    brief_id: int,
    format: str,
    include_research: bool = Query(False, description="Include research results"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export brief in various formats (txt, json, markdown)
    """
    try:
        # Validate brief access
        brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        if format not in ['txt', 'json', 'markdown', 'md']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported export format. Use: txt, json, markdown, md"
            )
        
        # Prepare export data
        export_data = {
            "title": brief.title,
            "description": brief.description,
            "case_type": brief.case_type,
            "jurisdiction": brief.jurisdiction,
            "court_level": brief.court_level,
            "content": brief.content,
            "word_count": brief.word_count,
            "status": brief.status.value,
            "created_at": brief.created_at.isoformat(),
            "updated_at": brief.updated_at.isoformat() if brief.updated_at else None
        }
        
        if include_research:
            research_results = db.query(ResearchResult).filter(
                ResearchResult.brief_id == brief_id
            ).all()
            
            export_data["research_results"] = [
                {
                    "query": r.query,
                    "title": r.title,
                    "citation": r.citation,
                    "summary": r.summary,
                    "relevance_score": r.relevance_score
                } for r in research_results
            ]
        
        # Format based on requested type
        if format == 'json':
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=export_data,
                headers={"Content-Disposition": f"attachment; filename=brief_{brief_id}.json"}
            )
        
        elif format in ['markdown', 'md']:
            content = f"# {export_data['title']}\n\n"
            if export_data['description']:
                content += f"**Description:** {export_data['description']}\n\n"
            
            content += f"**Case Type:** {export_data['case_type'] or 'N/A'}\n"
            content += f"**Jurisdiction:** {export_data['jurisdiction'] or 'N/A'}\n"
            content += f"**Court Level:** {export_data['court_level'] or 'N/A'}\n"
            content += f"**Status:** {export_data['status']}\n"
            content += f"**Word Count:** {export_data['word_count']}\n\n"
            
            content += "## Brief Content\n\n"
            content += export_data['content'] or "No content available"
            
            if include_research and 'research_results' in export_data:
                content += "\n\n## Research Results\n\n"
                for i, result in enumerate(export_data['research_results'], 1):
                    content += f"### {i}. {result['title'] or 'Untitled Research'}\n\n"
                    content += f"**Citation:** {result['citation'] or 'N/A'}\n"
                    content += f"**Relevance Score:** {result['relevance_score'] or 'N/A'}/10\n"
                    if result['summary']:
                        content += f"**Summary:** {result['summary']}\n"
                    content += "\n---\n\n"
            
            from fastapi.responses import Response
            return Response(
                content=content,
                media_type="text/markdown",
                headers={"Content-Disposition": f"attachment; filename=brief_{brief_id}.md"}
            )
            
        else:  # txt format
            content = f"LEGAL BRIEF: {export_data['title']}\n"
            content += "=" * (len(export_data['title']) + 14) + "\n\n"
            
            if export_data['description']:
                content += f"Description: {export_data['description']}\n"
            
            content += f"Case Type: {export_data['case_type'] or 'N/A'}\n"
            content += f"Jurisdiction: {export_data['jurisdiction'] or 'N/A'}\n"
            content += f"Court Level: {export_data['court_level'] or 'N/A'}\n"
            content += f"Status: {export_data['status']}\n"
            content += f"Word Count: {export_data['word_count']}\n"
            content += f"Created: {export_data['created_at']}\n\n"
            
            content += "CONTENT:\n"
            content += "-" * 50 + "\n"
            content += export_data['content'] or "No content available"
            
            if include_research and 'research_results' in export_data:
                content += "\n\n" + "=" * 50 + "\n"
                content += "RESEARCH RESULTS:\n"
                content += "=" * 50 + "\n\n"
                
                for i, result in enumerate(export_data['research_results'], 1):
                    content += f"{i}. {result['title'] or 'Untitled Research'}\n"
                    content += f"   Citation: {result['citation'] or 'N/A'}\n"
                    content += f"   Relevance: {result['relevance_score'] or 'N/A'}/10\n"
                    if result['summary']:
                        content += f"   Summary: {result['summary']}\n"
                    content += "\n" + "-" * 30 + "\n\n"
            
            from fastapi.responses import Response
            return Response(
                content=content,
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=brief_{brief_id}.txt"}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export brief error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export brief"
        )

# Advanced search endpoint
@router.post("/search", response_model=List[schemas.BriefSummary])
async def advanced_brief_search(
    search_request: schemas.BriefSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Advanced search for briefs with multiple criteria
    """
    try:
        # Build base query
        query = db.query(Brief).filter(Brief.owner_id == current_user.id)
        
        # Apply filters from search request
        if search_request.query:
            search_term = f"%{search_request.query}%"
            query = query.filter(
                or_(
                    Brief.title.ilike(search_term),
                    Brief.description.ilike(search_term),
                    Brief.content.ilike(search_term),
                    Brief.research_notes.ilike(search_term)
                )
            )
        
        if search_request.case_type:
            query = query.filter(Brief.case_type.ilike(f"%{search_request.case_type}%"))
        
        if search_request.status:
            query = query.filter(Brief.status == search_request.status)
        
        if search_request.jurisdiction:
            query = query.filter(Brief.jurisdiction.ilike(f"%{search_request.jurisdiction}%"))
        
        if search_request.date_from:
            query = query.filter(Brief.created_at >= search_request.date_from)
        
        if search_request.date_to:
            query = query.filter(Brief.created_at <= search_request.date_to)
        
        if search_request.owner_id and current_user.is_admin:
            # Only admins can search other users' briefs
            query = query.filter(Brief.owner_id == search_request.owner_id)
        
        # Execute query with pagination
        briefs = query.order_by(desc(Brief.updated_at)).offset(search_request.offset).limit(search_request.limit).all()
        
        # Convert to summary format
        brief_summaries = []
        for brief in briefs:
            brief_summaries.append(schemas.BriefSummary(
                id=brief.id,
                title=brief.title,
                case_type=brief.case_type,
                status=brief.status,
                word_count=brief.word_count,
                created_at=brief.created_at,
                updated_at=brief.updated_at
            ))
        
        return brief_summaries
        
    except Exception as e:
        logger.error(f"Advanced search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform advanced search"
        )

# Bulk operations
@router.post("/bulk/update-status")
async def bulk_update_status(
    brief_ids: List[int],
    new_status: schemas.BriefStatusEnum,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update status for multiple briefs
    """
    try:
        # Validate all briefs belong to current user
        briefs = db.query(Brief).filter(
            and_(
                Brief.id.in_(brief_ids),
                Brief.owner_id == current_user.id
            )
        ).all()
        
        if len(briefs) != len(brief_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Some briefs not found or access denied"
            )
        
        # Update all briefs
        updated_count = 0
        for brief in briefs:
            brief.status = new_status
            brief.updated_at = datetime.now()
            updated_count += 1
        
        db.commit()
        
        logger.info(f"Bulk updated {updated_count} briefs to status {new_status.value}")
        
        return {
            "message": f"Successfully updated {updated_count} briefs",
            "updated_count": updated_count,
            "new_status": new_status.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk update error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform bulk update"
        )

@router.delete("/bulk/archive")
async def bulk_archive_briefs(
    brief_ids: List[int],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Archive multiple briefs
    """
    try:
        # Validate all briefs belong to current user
        briefs = db.query(Brief).filter(
            and_(
                Brief.id.in_(brief_ids),
                Brief.owner_id == current_user.id
            )
        ).all()
        
        if len(briefs) != len(brief_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Some briefs not found or access denied"
            )
        
        # Archive all briefs
        archived_count = 0
        for brief in briefs:
            brief.status = BriefStatus.ARCHIVED
            brief.updated_at = datetime.now()
            archived_count += 1
        
        db.commit()
        
        logger.info(f"Bulk archived {archived_count} briefs")
        
        return {
            "message": f"Successfully archived {archived_count} briefs",
            "archived_count": archived_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk archive error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform bulk archive"
        )

# Template and collaboration features
@router.post("/{brief_id}/duplicate", response_model=schemas.BriefResponse)
async def duplicate_brief(
    brief_id: int,
    new_title: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a duplicate of an existing brief
    """
    try:
        # Get original brief
        original_brief = db.query(Brief).filter(
            and_(Brief.id == brief_id, Brief.owner_id == current_user.id)
        ).first()
        
        if not original_brief:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Brief not found"
            )
        
        # Create duplicate
        duplicate_title = new_title or f"Copy of {original_brief.title}"
        
        duplicate_brief = Brief(
            title=duplicate_title,
            description=original_brief.description,
            case_type=original_brief.case_type,
            jurisdiction=original_brief.jurisdiction,
            court_level=original_brief.court_level,
            content=original_brief.content,
            research_notes=original_brief.research_notes,
            citations=original_brief.citations,
            status=BriefStatus.DRAFT,
            owner_id=current_user.id,
            word_count=original_brief.word_count,
            page_count=original_brief.page_count
        )
        
        db.add(duplicate_brief)
        db.commit()
        db.refresh(duplicate_brief)
        
        # Copy research results
        original_research = db.query(ResearchResult).filter(
            ResearchResult.brief_id == brief_id
        ).all()
        
        for research in original_research:
            duplicate_research = ResearchResult(
                brief_id=duplicate_brief.id,
                query=research.query,
                source_type=research.source_type,
                title=research.title,
                citation=research.citation,
                court=research.court,
                year=research.year,
                summary=research.summary,
                relevant_excerpt=research.relevant_excerpt,
                relevance_score=research.relevance_score,
                agent_notes=research.agent_notes,
                found_by_agent=research.found_by_agent,
                verified=research.verified
            )
            db.add(duplicate_research)
        
        db.commit()
        
        logger.info(f"Duplicated brief {brief_id} to {duplicate_brief.id}")
        
        return duplicate_brief
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Duplicate brief error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to duplicate brief"
        )