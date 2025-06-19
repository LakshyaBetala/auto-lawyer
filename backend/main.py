"""
AutoLawyer FastAPI Main Application - Part 1
Entry point for the legal brief drafting system with local LLM integration
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

# Database imports
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

# Internal imports
from db.database import engine, SessionLocal, get_db, init_db
from db.models import Base, User, Brief, ModelConfig
from db import schemas
from config import settings, logger
from local_llm.llm_runner import LLMRunner, LLMManager
from core.agent_executor import AgentExecutor
from core.graph_builder import WorkflowGraph

# Route imports
from routes import users, briefs, agents

# Security and utilities
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

# Global variables for application state
llm_manager: LLMManager = None
agent_executor: AgentExecutor = None
workflow_graph: WorkflowGraph = None

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("autolawyer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown events
    """
    logger.info("🚀 Starting AutoLawyer application...")
    
    # Startup
    try:
        # Initialize database
        logger.info("📦 Initializing database...")
        await init_database()
        
        # Initialize local LLM manager
        logger.info("🧠 Initializing local LLM manager...")
        await init_llm_manager()
        
        # Initialize agent system
        logger.info("🤖 Initializing agent executor...")
        await init_agent_system()
        
        # Validate system health
        logger.info("🔍 Running system health checks...")
        await validate_system_health()
        
        logger.info("✅ AutoLawyer started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start AutoLawyer: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("🛑 Shutting down AutoLawyer...")
    try:
        # Cleanup LLM resources
        if llm_manager:
            await llm_manager.cleanup()
            logger.info("🧠 LLM manager cleaned up")
        
        # Close database connections
        engine.dispose()
        logger.info("📦 Database connections closed")
        
        logger.info("✅ AutoLawyer shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")

# Create FastAPI application
app = FastAPI(
    title="AutoLawyer API",
    description="Local LLM-powered legal brief drafting system with multi-agent collaboration",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files if they exist
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")