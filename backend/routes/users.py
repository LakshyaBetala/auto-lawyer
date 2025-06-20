"""
AutoLawyer User Routes
FastAPI routes for user management, authentication, and administration
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks, Form
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import secrets
import uuid

# Internal imports
from db.database import get_db
from db.models import User, Brief, AgentLog, ModelConfig, BriefStatus, AgentRole
from db import schemas
from config import logger, settings

# Import from main for dependency injection
from main import get_current_user, get_current_admin_user, verify_password, get_password_hash, create_access_token

router = APIRouter()

# User Registration and Profile Management
@router.post("/register", response_model=schemas.UserResponse)
async def register_user(
    user_data: schemas.UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Register a new user account
    """
    try:
        # Check if username already exists
        existing_user = db.query(User).filter(User.username == user_data.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        existing_email = db.query(User).filter(User.email == user_data.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create new user
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            is_active=True,
            is_admin=False
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Log registration
        logger.info(f"New user registered: {new_user.username} (ID: {new_user.id})")
        
        # Send welcome email in background (if email service is configured)
        background_tasks.add_task(send_welcome_email, new_user.email, new_user.full_name)
        
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )

@router.get("/profile", response_model=schemas.UserResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user's profile information
    """
    return current_user

@router.put("/profile", response_model=schemas.UserResponse)
async def update_user_profile(
    user_update: schemas.UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update current user's profile information
    """
    try:
        # Check if username is being changed and if it's available
        if user_update.username and user_update.username != current_user.username:
            existing_user = db.query(User).filter(
                and_(
                    User.username == user_update.username,
                    User.id != current_user.id
                )
            ).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )
        
        # Check if email is being changed and if it's available
        if user_update.email and user_update.email != current_user.email:
            existing_email = db.query(User).filter(
                and_(
                    User.email == user_update.email,
                    User.id != current_user.id
                )
            ).first()
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already taken"
                )
        
        # Update fields
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(current_user, field) and field != 'is_active':  # Prevent self-deactivation
                setattr(current_user, field, value)
        
        current_user.updated_at = datetime.now()
        
        db.commit()
        db.refresh(current_user)
        
        logger.info(f"User {current_user.id} updated profile")
        
        return current_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@router.post("/change-password")
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user's password
    """
    try:
        # Verify current password
        if not verify_password(current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid current password"
            )
        
        # Validate new password strength (basic validation)
        if len(new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters long"
            )
        
        # Hash new password
        hashed_new_password = get_password_hash(new_password)
        
        # Update password
        current_user.hashed_password = hashed_new_password
        current_user.updated_at = datetime.now()
        
        db.commit()
        
        logger.info(f"User {current_user.id} changed password")
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

# User Activity and Statistics
@router.get("/activity", response_model=Dict[str, Any])
async def get_user_activity(
    days_back: int = Query(30, ge=1, le=365, description="Number of days to include"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's activity statistics and recent actions
    """
    try:
        since_date = datetime.now() - timedelta(days=days_back)
        
        # Brief statistics
        brief_stats = db.query(
            func.count(Brief.id).label('total_briefs'),
            func.count(func.nullif(Brief.status == BriefStatus.COMPLETED, False)).label('completed_briefs'),
            func.count(func.nullif(Brief.status == BriefStatus.IN_PROGRESS, False)).label('active_briefs'),
            func.sum(Brief.word_count).label('total_words'),
            func.avg(Brief.word_count).label('avg_words_per_brief')
        ).filter(
            and_(
                Brief.owner_id == current_user.id,
                Brief.created_at >= since_date
            )
        ).first()
        
        # Agent execution statistics
        agent_stats = db.query(
            AgentLog.agent_role,
            func.count(AgentLog.id).label('executions'),
            func.avg(AgentLog.execution_time_ms).label('avg_execution_time'),
            func.sum(AgentLog.tokens_used).label('total_tokens')
        ).filter(
            and_(
                AgentLog.user_id == current_user.id,
                AgentLog.created_at >= since_date
            )
        ).group_by(AgentLog.agent_role).all()
        
        # Recent activity (last 10 actions)
        recent_activity = db.query(
            AgentLog.created_at,
            AgentLog.agent_role,
            AgentLog.action,
            Brief.title.label('brief_title')
        ).join(Brief, AgentLog.brief_id == Brief.id).filter(
            AgentLog.user_id == current_user.id
        ).order_by(desc(AgentLog.created_at)).limit(10).all()
        
        # Usage by day (for the last 7 days)
        daily_usage = db.query(
            func.date(AgentLog.created_at).label('date'),
            func.count(AgentLog.id).label('executions'),
            func.sum(AgentLog.tokens_used).label('tokens')
        ).filter(
            and_(
                AgentLog.user_id == current_user.id,
                AgentLog.created_at >= datetime.now() - timedelta(days=7)
            )
        ).group_by(func.date(AgentLog.created_at)).all()
        
        return {
            "period_days": days_back,
            "brief_statistics": {
                "total_briefs": brief_stats.total_briefs or 0,
                "completed_briefs": brief_stats.completed_briefs or 0,
                "active_briefs": brief_stats.active_briefs or 0,
                "total_words": int(brief_stats.total_words or 0),
                "avg_words_per_brief": float(brief_stats.avg_words_per_brief or 0)
            },
            "agent_statistics": [
                {
                    "agent_role": stat.agent_role.value,
                    "executions": stat.executions,
                    "avg_execution_time_ms": float(stat.avg_execution_time or 0),
                    "total_tokens": int(stat.total_tokens or 0)
                } for stat in agent_stats
            ],
            "recent_activity": [
                {
                    "timestamp": activity.created_at,
                    "agent_role": activity.agent_role.value,
                    "action": activity.action,
                    "brief_title": activity.brief_title
                } for activity in recent_activity
            ],
            "daily_usage": [
                {
                    "date": usage.date.isoformat() if usage.date else None,
                    "executions": usage.executions,
                    "tokens": int(usage.tokens or 0)
                } for usage in daily_usage
            ]
        }
        
    except Exception as e:
        logger.error(f"Get user activity error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user activity"
        )

@router.get("/dashboard", response_model=Dict[str, Any])
async def get_user_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user dashboard data with quick stats and recent items
    """
    try:
        # Quick statistics
        total_briefs = db.query(Brief).filter(Brief.owner_id == current_user.id).count()
        draft_briefs = db.query(Brief).filter(
            and_(Brief.owner_id == current_user.id, Brief.status == BriefStatus.DRAFT)
        ).count()
        active_briefs = db.query(Brief).filter(
            and_(Brief.owner_id == current_user.id, Brief.status == BriefStatus.IN_PROGRESS)
        ).count()
        
        # Recent briefs (last 5)
        recent_briefs = db.query(Brief).filter(
            Brief.owner_id == current_user.id
        ).order_by(desc(Brief.updated_at)).limit(5).all()
        
        # Recent agent executions (last 5)
        recent_executions = db.query(
            AgentLog.created_at,
            AgentLog.agent_role,
            AgentLog.action,
            AgentLog.success,
            Brief.title.label('brief_title')
        ).join(Brief, AgentLog.brief_id == Brief.id).filter(
            AgentLog.user_id == current_user.id
        ).order_by(desc(AgentLog.created_at)).limit(5).all()
        
        # Usage this week
        week_start = datetime.now() - timedelta(days=7)
        week_executions = db.query(AgentLog).filter(
            and_(
                AgentLog.user_id == current_user.id,
                AgentLog.created_at >= week_start
            )
        ).count()
        
        week_tokens = db.query(func.sum(AgentLog.tokens_used)).filter(
            and_(
                AgentLog.user_id == current_user.id,
                AgentLog.created_at >= week_start
            )
        ).scalar() or 0
        
        return {
            "user_info": {
                "username": current_user.username,
                "full_name": current_user.full_name,
                "email": current_user.email,
                "member_since": current_user.created_at,
                "last_active": current_user.updated_at
            },
            "quick_stats": {
                "total_briefs": total_briefs,
                "draft_briefs": draft_briefs,
                "active_briefs": active_briefs,
                "week_executions": week_executions,
                "week_tokens": int(week_tokens)
            },
            "recent_briefs": [
                {
                    "id": brief.id,
                    "title": brief.title,
                    "status": brief.status.value,
                    "updated_at": brief.updated_at,
                    "word_count": brief.word_count
                } for brief in recent_briefs
            ],
            "recent_executions": [
                {
                    "timestamp": execution.created_at,
                    "agent_role": execution.agent_role.value,
                    "action": execution.action,
                    "success": execution.success,
                    "brief_title": execution.brief_title
                } for execution in recent_executions
            ]
        }
        
    except Exception as e:
        logger.error(f"Get user dashboard error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard data"
        )

# Admin User Management
@router.get("/", response_model=List[schemas.UserResponse])
async def list_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(50, ge=1, le=200, description="Number of users to return"),
    search: Optional[str] = Query(None, description="Search in username, email, or full name"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    is_admin: Optional[bool] = Query(None, description="Filter by admin status"),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    List all users (admin only)
    """
    try:
        # Build query
        query = db.query(User)
        
        # Apply filters
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    User.username.ilike(search_term),
                    User.email.ilike(search_term),
                    User.full_name.ilike(search_term)
                )
            )
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        if is_admin is not None:
            query = query.filter(User.is_admin == is_admin)
        
        # Execute query with pagination
        users = query.order_by(desc(User.created_at)).offset(skip).limit(limit).all()
        
        return users
        
    except Exception as e:
        logger.error(f"List users error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )

@router.get("/{user_id}", response_model=schemas.UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get specific user by ID (admin only)
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )

@router.put("/{user_id}", response_model=schemas.UserResponse)
async def update_user(
    user_id: int,
    user_update: schemas.UserUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Update user information (admin only)
    """
    try:
        # Get user to update
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent admin from deactivating themselves
        if user_id == current_user.id and user_update.is_active is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )
        
        # Check for username conflicts
        if user_update.username and user_update.username != user.username:
            existing_user = db.query(User).filter(
                and_(
                    User.username == user_update.username,
                    User.id != user_id
                )
            ).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )
        
        # Check for email conflicts
        if user_update.email and user_update.email != user.email:
            existing_email = db.query(User).filter(
                and_(
                    User.email == user_update.email,
                    User.id != user_id
                )
            ).first()
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already taken"
                )
        
        # Update fields
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        user.updated_at = datetime.now()
        
        db.commit()
        db.refresh(user)
        
        logger.info(f"Admin {current_user.id} updated user {user_id}")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Deactivate user account (admin only)
    """
    try:
        # Get user to deactivate
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent admin from deleting themselves
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Soft delete by deactivating
        user.is_active = False
        user.updated_at = datetime.now()
        
        db.commit()
        
        logger.info(f"Admin {current_user.id} deactivated user {user_id}")
        
        return {"message": "User account deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate user"
        )

@router.post("/{user_id}/reset-password")
async def admin_reset_password(
    user_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Reset user password and send new temporary password (admin only)
    """
    try:
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Generate temporary password
        temp_password = secrets.token_urlsafe(12)
        hashed_temp_password = get_password_hash(temp_password)
        
        # Update user password
        user.hashed_password = hashed_temp_password
        user.updated_at = datetime.now()
        
        db.commit()
        
        # Send new password via email in background
        background_tasks.add_task(
            send_password_reset_email,
            user.email,
            user.full_name,
            temp_password
        )
        
        logger.info(f"Admin {current_user.id} reset password for user {user_id}")
        
        return {
            "message": "Password reset successfully. New temporary password sent to user's email.",
            "temporary_password": temp_password  # In production, don't return this
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin password reset error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset user password"
        )

# User Analytics (Admin)
@router.get("/{user_id}/analytics")
async def get_user_analytics(
    user_id: int,
    days_back: int = Query(30, ge=1, le=365, description="Number of days to include"),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed analytics for a specific user (admin only)
    """
    try:
        # Verify user exists
        target_user = db.query(User).filter(User.id == user_id).first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        since_date = datetime.now() - timedelta(days=days_back)
        
        # Brief analytics
        brief_analytics = db.query(
            func.count(Brief.id).label('total_briefs'),
            func.count(func.nullif(Brief.status == BriefStatus.COMPLETED, False)).label('completed'),
            func.count(func.nullif(Brief.status == BriefStatus.IN_PROGRESS, False)).label('in_progress'),
            func.count(func.nullif(Brief.status == BriefStatus.DRAFT, False)).label('drafts'),
            func.sum(Brief.word_count).label('total_words'),
            func.avg(Brief.word_count).label('avg_words')
        ).filter(
            and_(
                Brief.owner_id == user_id,
                Brief.created_at >= since_date
            )
        ).first()
        
        # Agent usage analytics
        agent_analytics = db.query(
            AgentLog.agent_role,
            func.count(AgentLog.id).label('executions'),
            func.avg(AgentLog.execution_time_ms).label('avg_time'),
            func.sum(AgentLog.tokens_used).label('total_tokens'),
            func.avg(func.cast(AgentLog.success, db.Integer)).label('success_rate')
        ).filter(
            and_(
                AgentLog.user_id == user_id,
                AgentLog.created_at >= since_date
            )
        ).group_by(AgentLog.agent_role).all()
        
        # Usage timeline (daily for last 30 days)
        timeline_data = db.query(
            func.date(AgentLog.created_at).label('date'),
            func.count(AgentLog.id).label('executions'),
            func.sum(AgentLog.tokens_used).label('tokens')
        ).filter(
            and_(
                AgentLog.user_id == user_id,
                AgentLog.created_at >= datetime.now() - timedelta(days=30)
            )
        ).group_by(func.date(AgentLog.created_at)).all()
        
        return {
            "user_info": {
                "id": target_user.id,
                "username": target_user.username,
                "full_name": target_user.full_name,
                "email": target_user.email,
                "created_at": target_user.created_at,
                "is_active": target_user.is_active
            },
            "period_days": days_back,
            "brief_analytics": {
                "total_briefs": brief_analytics.total_briefs or 0,
                "completed": brief_analytics.completed or 0,
                "in_progress": brief_analytics.in_progress or 0,
                "drafts": brief_analytics.drafts or 0,
                "total_words": int(brief_analytics.total_words or 0),
                "avg_words": float(brief_analytics.avg_words or 0)
            },
            "agent_analytics": [
                {
                    "agent_role": analytics.agent_role.value,
                    "executions": analytics.executions,
                    "avg_execution_time_ms": float(analytics.avg_time or 0),
                    "total_tokens": int(analytics.total_tokens or 0),
                    "success_rate": float(analytics.success_rate or 0)
                } for analytics in agent_analytics
            ],
            "usage_timeline": [
                {
                    "date": timeline.date.isoformat() if timeline.date else None,
                    "executions": timeline.executions,
                    "tokens": int(timeline.tokens or 0)
                } for timeline in timeline_data
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user analytics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user analytics"
        )

# System-wide user statistics (Admin)
@router.get("/statistics/system", response_model=Dict[str, Any])
async def get_system_user_statistics(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get system-wide user statistics (admin only)
    """
    try:
        # User counts
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        admin_users = db.query(User).filter(User.is_admin == True).count()
        
        # Users by registration date (last 30 days)
        registration_data = db.query(
            func.date(User.created_at).label('date'),
            func.count(User.id).label('registrations')
        ).filter(
            User.created_at >= datetime.now() - timedelta(days=30)
        ).group_by(func.date(User.created_at)).all()
        
        # Most active users (by agent executions in last 30 days)
        most_active = db.query(
            User.username,
            User.full_name,
            func.count(AgentLog.id).label('executions')
        ).join(AgentLog, User.id == AgentLog.user_id).filter(
            AgentLog.created_at >= datetime.now() - timedelta(days=30)
        ).group_by(User.id, User.username, User.full_name).order_by(
            desc(func.count(AgentLog.id))
        ).limit(10).all()
        
        # Overall system usage
        total_executions = db.query(AgentLog).count()
        total_briefs = db.query(Brief).count()
        total_tokens = db.query(func.sum(AgentLog.tokens_used)).scalar() or 0
        
        return {
            "user_counts": {
                "total_users": total_users,
                "active_users": active_users,
                "inactive_users": total_users - active_users,
                "admin_users": admin_users
            },
            "registration_timeline": [
                {
                    "date": reg.date.isoformat() if reg.date else None,
                    "registrations": reg.registrations
                } for reg in registration_data
            ],
            "most_active_users": [
                {
                    "username": user.username,
                    "full_name": user.full_name,
                    "executions": user.executions
                } for user in most_active
            ],
            "system_usage": {
                "total_executions": total_executions,
                "total_briefs": total_briefs,
                "total_tokens": int(total_tokens)
            }
        }
        
    except Exception as e:
        logger.error(f"Get system user statistics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system user statistics"
        )

# Bulk user operations (Admin)
@router.post("/bulk/deactivate")
async def bulk_deactivate_users(
    user_ids: List[int],
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Deactivate multiple users (admin only)
    """
    try:
        # Prevent admin from deactivating themselves
        if current_user.id in user_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )
        
        # Get users to deactivate
        users = db.query(User).filter(User.id.in_(user_ids)).all()
        
        if len(users) != len(user_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Some users not found"
            )
        
        # Deactivate users
        deactivated_count = 0
        for user in users:
            user.is_active = False
            user.updated_at = datetime.now()
            deactivated_count += 1
        
        db.commit()
        
        logger.info(f"Admin {current_user.id} bulk deactivated {deactivated_count} users")
        
        return {
            "message": f"Successfully deactivated {deactivated_count} users",
            "deactivated_count": deactivated_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk deactivate users error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate users"
        )

@router.post("/bulk/activate")
async def bulk_activate_users(
    user_ids: List[int],
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Activate multiple users (admin only)
    """
    try:
        # Get users to activate
        users = db.query(User).filter(User.id.in_(user_ids)).all()
        
        if len(users) != len(user_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Some users not found"
            )
        
        # Activate users
        activated_count = 0
        for user in users:
            user.is_active = True
            user.updated_at = datetime.now()
            activated_count += 1
        
        db.commit()
        
        logger.info(f"Admin {current_user.id} bulk activated {activated_count} users")
        
        return {
            "message": f"Successfully activated {activated_count} users",
            "activated_count": activated_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk activate users error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate users"
        )

# Account recovery endpoints
@router.post("/forgot-password")
async def forgot_password(
    email: str = Form(...),
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Initiate password reset process
    """
    try:
        # Find user by email
        user = db.query(User).filter(User.email == email).first()
        
        # Always return success to prevent email enumeration
        if user and user.is_active:
            # Generate reset token
            reset_token = str(uuid.uuid4())
            
            # In production, store reset token in database with expiration
            # For now, we'll just send email with token
            background_tasks.add_task(
                send_password_reset_email,
                user.email,
                user.full_name,
                reset_token
            )
            
            logger.info(f"Password reset requested for user {user.id}")
        
        return {
            "message": "If an account with that email exists, a password reset link has been sent."
        }
        
    except Exception as e:
        logger.error(f"Forgot password error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process password reset request"
        )

# User preferences and settings
@router.get("/preferences")
async def get_user_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user preferences and settings
    """
    try:
        # Get preferred models for each agent type
        preferred_models = {}
        
        for agent_role in AgentRole:
            preferred_model = db.query(ModelConfig).filter(
                getattr(ModelConfig, f'preferred_for_{agent_role.value}') == True,
                ModelConfig.is_active == True
            ).first()
            
            preferred_models[agent_role.value] = {
                "model_name": preferred_model.name if preferred_model else None,
                "model_id": preferred_model.id if preferred_model else None
            }
        
        # Get available models
        available_models = db.query(ModelConfig).filter(ModelConfig.is_active == True).all()
        
        return {
            "user_id": current_user.id,
            "preferred_models": preferred_models,
            "available_models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "model_type": model.model_type,
                    "context_length": model.context_length
                } for model in available_models
            ],
            "default_settings": {
                "auto_save_drafts": True,
                "email_notifications": True,
                "theme": "light"
            }
        }
        
    except Exception as e:
        logger.error(f"Get user preferences error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user preferences"
        )

@router.put("/preferences")
async def update_user_preferences(
    preferences: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user preferences and settings
    """
    try:
        # Update model preferences
        if "preferred_models" in preferences:
            for agent_role, model_info in preferences["preferred_models"].items():
                if model_info.get("model_id"):
                    # Verify model exists and is active
                    model = db.query(ModelConfig).filter(
                        and_(
                            ModelConfig.id == model_info["model_id"],
                            ModelConfig.is_active == True
                        )
                    ).first()
                    
                    if model:
                        # Reset all models for this agent type
                        db.query(ModelConfig).filter(
                            getattr(ModelConfig, f'preferred_for_{agent_role}') == True
                        ).update({f'preferred_for_{agent_role}': False})
                        
                        # Set new preferred model
                        setattr(model, f'preferred_for_{agent_role}', True)
        
        db.commit()
        
        logger.info(f"User {current_user.id} updated preferences")
        
        return {"message": "Preferences updated successfully"}
        
    except Exception as e:
        logger.error(f"Update user preferences error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user preferences"
        )

# Export user data (GDPR compliance)
@router.get("/export-data")
async def export_user_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export all user data for GDPR compliance
    """
    try:
        # Get user's briefs
        briefs = db.query(Brief).filter(Brief.owner_id == current_user.id).all()
        
        # Get user's agent logs
        agent_logs = db.query(AgentLog).filter(AgentLog.user_id == current_user.id).all()
        
        # Compile all user data
        user_data = {
            "user_profile": {
                "id": current_user.id,
                "username": current_user.username,
                "email": current_user.email,
                "full_name": current_user.full_name,
                "created_at": current_user.created_at.isoformat(),
                "updated_at": current_user.updated_at.isoformat() if current_user.updated_at else None,
                "is_active": current_user.is_active,
                "is_admin": current_user.is_admin
            },
            "briefs": [
                {
                    "id": brief.id,
                    "title": brief.title,
                    "description": brief.description,
                    "case_type": brief.case_type,
                    "jurisdiction": brief.jurisdiction,
                    "court_level": brief.court_level,
                    "content": brief.content,
                    "status": brief.status.value,
                    "word_count": brief.word_count,
                    "created_at": brief.created_at.isoformat(),
                    "updated_at": brief.updated_at.isoformat() if brief.updated_at else None
                } for brief in briefs
            ],
            "agent_activity": [
                {
                    "id": log.id,
                    "brief_id": log.brief_id,
                    "agent_role": log.agent_role.value,
                    "action": log.action,
                    "execution_time_ms": log.execution_time_ms,
                    "tokens_used": log.tokens_used,
                    "success": log.success,
                    "created_at": log.created_at.isoformat()
                } for log in agent_logs
            ],
            "export_metadata": {
                "export_date": datetime.now().isoformat(),
                "export_format": "json",
                "total_briefs": len(briefs),
                "total_agent_executions": len(agent_logs)
            }
        }
        
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=user_data,
            headers={
                "Content-Disposition": f"attachment; filename=autolawyer_user_data_{current_user.id}.json"
            }
        )
        
    except Exception as e:
        logger.error(f"Export user data error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export user data"
        )

# Background task functions
async def send_welcome_email(email: str, full_name: str):
    """
    Send welcome email to new user (placeholder implementation)
    """
    try:
        # In production, implement actual email sending
        logger.info(f"Welcome email would be sent to {email} ({full_name})")
        
        # Example email content:
        # - Welcome message
        # - Getting started guide
        # - Support contact information
        
    except Exception as e:
        logger.error(f"Failed to send welcome email: {str(e)}")

async def send_password_reset_email(email: str, full_name: str, reset_token_or_password: str):
    """
    Send password reset email (placeholder implementation)
    """
    try:
        # In production, implement actual email sending
        logger.info(f"Password reset email would be sent to {email} ({full_name})")
        
        # Example email content:
        # - Password reset instructions
        # - Reset link or temporary password
        # - Security tips
        
    except Exception as e:
        logger.error(f"Failed to send password reset email: {str(e)}")

# Account deletion (GDPR right to be forgotten)
@router.delete("/delete-account")
async def delete_user_account(
    password: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete user account and all associated data (GDPR compliance)
    """
    try:
        # Verify password
        if not verify_password(password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid password"
            )
        
        # Prevent admin from deleting themselves if they're the only admin
        if current_user.is_admin:
            admin_count = db.query(User).filter(
                and_(User.is_admin == True, User.is_active == True)
            ).count()
            
            if admin_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete the last admin account"
                )
        
        # Delete all user data
        # Note: Due to foreign key constraints, delete in correct order
        
        # Delete agent logs
        db.query(AgentLog).filter(AgentLog.user_id == current_user.id).delete()
        
        # Delete research results (through briefs)
        from db.models import ResearchResult, BriefVersion
        brief_ids = [brief.id for brief in db.query(Brief).filter(Brief.owner_id == current_user.id).all()]
        
        if brief_ids:
            db.query(ResearchResult).filter(ResearchResult.brief_id.in_(brief_ids)).delete()
            db.query(BriefVersion).filter(BriefVersion.brief_id.in_(brief_ids)).delete()
        
        # Delete briefs
        db.query(Brief).filter(Brief.owner_id == current_user.id).delete()
        
        # Delete user
        db.delete(current_user)
        db.commit()
        
        logger.info(f"User account {current_user.id} deleted successfully")
        
        return {"message": "Account deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user account error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )