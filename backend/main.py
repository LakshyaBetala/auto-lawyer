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
    
# Initialization functions
async def init_database():
    """Initialize database tables and default data"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("📊 Database tables created/verified")
        
        # Test database connection
        with SessionLocal() as db:
            result = db.execute(text("SELECT 1")).scalar()
            if result != 1:
                raise Exception("Database connection test failed")
        
        logger.info("✅ Database connection verified")
        
        # Create default admin user if not exists
        await create_default_admin()
        
        # Initialize default model configurations
        await init_default_models()
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        raise

async def create_default_admin():
    """Create default admin user if none exists"""
    try:
        with SessionLocal() as db:
            admin_exists = db.query(User).filter(User.is_admin == True).first()
            if not admin_exists:
                hashed_password = pwd_context.hash(settings.DEFAULT_ADMIN_PASSWORD)
                admin_user = User(
                    username="admin",
                    email="admin@autolawyer.local",
                    full_name="System Administrator",
                    hashed_password=hashed_password,
                    is_admin=True,
                    is_active=True
                )
                db.add(admin_user)
                db.commit()
                logger.info("👤 Default admin user created")
            else:
                logger.info("👤 Admin user already exists")
    except Exception as e:
        logger.error(f"❌ Failed to create admin user: {str(e)}")
        raise

async def init_default_models():
    """Initialize default LLM model configurations"""
    try:
        with SessionLocal() as db:
            model_count = db.query(ModelConfig).count()
            if model_count == 0:
                # Add default model config (user needs to update paths)
                default_model = ModelConfig(
                    name="default-local-model",
                    model_path="models/default-model.gguf",
                    model_type="llama",
                    context_length=4096,
                    max_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    n_threads=4,
                    n_gpu_layers=0,
                    is_active=False,  # User needs to configure proper path
                    is_default=True,
                    preferred_for_research=True,
                    preferred_for_drafting=True,
                    preferred_for_summarizing=True
                )
                db.add(default_model)
                db.commit()
                logger.info("🤖 Default model configuration created")
            else:
                logger.info("🤖 Model configurations already exist")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model configs: {str(e)}")
        raise

async def init_llm_manager():
    """Initialize the local LLM manager"""
    global llm_manager
    try:
        llm_manager = LLMManager()
        await llm_manager.initialize()
        
        # Load active models
        with SessionLocal() as db:
            active_models = db.query(ModelConfig).filter(ModelConfig.is_active == True).all()
            for model_config in active_models:
                try:
                    await llm_manager.load_model(
                        name=model_config.name,
                        model_path=model_config.model_path,
                        config=model_config
                    )
                    logger.info(f"🧠 Loaded model: {model_config.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model {model_config.name}: {str(e)}")
        
        logger.info("✅ LLM manager initialized")
        
    except Exception as e:
        logger.error(f"❌ LLM manager initialization failed: {str(e)}")
        raise

async def init_agent_system():
    """Initialize the agent execution system"""
    global agent_executor, workflow_graph
    try:
        # Initialize agent executor
        agent_executor = AgentExecutor(llm_manager=llm_manager)
        await agent_executor.initialize()
        
        # Initialize workflow graph builder
        workflow_graph = WorkflowGraph(agent_executor=agent_executor)
        await workflow_graph.initialize()
        
        logger.info("✅ Agent system initialized")
        
    except Exception as e:
        logger.error(f"❌ Agent system initialization failed: {str(e)}")
        raise

async def validate_system_health():
    """Validate that all system components are working"""
    health_status = {
        "database": False,
        "llm_manager": False,
        "agent_system": False
    }
    
    try:
        # Test database
        with SessionLocal() as db:
            db.execute(text("SELECT 1")).scalar()
            health_status["database"] = True
        
        # Test LLM manager
        if llm_manager and llm_manager.has_active_models():
            health_status["llm_manager"] = True
        
        # Test agent system
        if agent_executor and workflow_graph:
            health_status["agent_system"] = True
        
        # Log status
        for component, status in health_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"{status_icon} {component}: {'OK' if status else 'FAILED'}")
        
        # Fail if critical components are down
        if not health_status["database"]:
            raise Exception("Database health check failed")
        
        if not health_status["llm_manager"]:
            logger.warning("⚠️ No active LLM models loaded - agents will not function")
        
    except Exception as e:
        logger.error(f"❌ System health validation failed: {str(e)}")
        raise

# Authentication functions
def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(db: SessionLocal = Depends(get_db), token_data: dict = Depends(verify_token)):
    """Get current user from token"""
    username = token_data.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    return user

def get_current_admin_user(current_user: User = Depends(get_current_user)):
    """Ensure current user is an admin"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)