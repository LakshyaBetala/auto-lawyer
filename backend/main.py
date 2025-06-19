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
# Include routers
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(briefs.router, prefix="/api/v1/briefs", tags=["briefs"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])

# Root and health endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs"""
    return RedirectResponse(url="/docs" if settings.DEBUG else "/health")

@app.get("/health", response_model=schemas.HealthCheckResponse, tags=["system"])
async def health_check():
    """System health check endpoint"""
    try:
        # Check database
        db_status = "ok"
        try:
            with SessionLocal() as db:
                db.execute(text("SELECT 1")).scalar()
        except Exception:
            db_status = "error"
        
        # Check LLM manager
        llm_status = "ok" if (llm_manager and llm_manager.has_active_models()) else "no_models"
        if not llm_manager:
            llm_status = "error"
        
        overall_status = "ok" if (db_status == "ok" and llm_status in ["ok", "no_models"]) else "error"
        
        return schemas.HealthCheckResponse(
            status=overall_status,
            database=db_status,
            local_llm=llm_status,
            timestamp=datetime.now(),
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return schemas.HealthCheckResponse(
            status="error",
            database="unknown",
            local_llm="unknown",
            timestamp=datetime.now(),
            version="1.0.0"
        )

@app.post("/api/v1/auth/login", response_model=Dict[str, Any], tags=["authentication"])
async def login(user_credentials: schemas.UserLogin, db: SessionLocal = Depends(get_db)):
    """User login endpoint"""
    try:
        # Find user
        user = db.query(User).filter(User.username == user_credentials.username).first()
        if not user or not verify_password(user_credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id, "is_admin": user.is_admin},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_admin": user.is_admin
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )

@app.get("/api/v1/system/info", tags=["system"])
async def system_info(current_user: User = Depends(get_current_admin_user)):
    """Get system information (admin only)"""
    try:
        with SessionLocal() as db:
            # Count statistics
            total_users = db.query(User).count()
            total_briefs = db.query(Brief).count()
            active_models = db.query(ModelConfig).filter(ModelConfig.is_active == True).count()
        
        # LLM manager info
        llm_info = {}
        if llm_manager:
            llm_info = {
                "loaded_models": list(llm_manager.get_loaded_models().keys()),
                "total_models": len(llm_manager.get_loaded_models()),
                "memory_usage": llm_manager.get_memory_usage() if hasattr(llm_manager, 'get_memory_usage') else "unknown"
            }
        
        return {
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "statistics": {
                "total_users": total_users,
                "total_briefs": total_briefs,
                "active_models": active_models
            },
            "llm_manager": llm_info,
            "uptime_seconds": (datetime.now() - datetime.now()).total_seconds()  # This would be tracked properly in production
        }
        
    except Exception as e:
        logger.error(f"System info error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system information"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(request, exc: SQLAlchemyError):
    """Handle database exceptions"""
    logger.error(f"Database error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Database error occurred",
            "detail": "Please try again later",
            "code": "DATABASE_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "code": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    )

# Development endpoints (only available in debug mode)
if settings.DEBUG:
    @app.get("/api/v1/dev/reset-db", tags=["development"])
    async def reset_database(current_user: User = Depends(get_current_admin_user)):
        """Reset database (development only)"""
        try:
            # Drop and recreate all tables
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            
            # Reinitialize default data
            await create_default_admin()
            await init_default_models()
            
            return {"message": "Database reset successfully"}
            
        except Exception as e:
            logger.error(f"Database reset error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset database"
            )
    
    @app.get("/api/v1/dev/logs", tags=["development"])
    async def get_logs(current_user: User = Depends(get_current_admin_user), lines: int = 100):
        """Get application logs (development only)"""
        try:
            log_file = Path("autolawyer.log")
            if not log_file.exists():
                return {"logs": [], "message": "No log file found"}
            
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            return {
                "logs": [line.strip() for line in recent_lines],
                "total_lines": len(all_lines),
                "returned_lines": len(recent_lines)
            }
            
        except Exception as e:
            logger.error(f"Get logs error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve logs"
            )

# Application entry point
if __name__ == "__main__":
    import uvicorn
    
    # Configuration based on environment
    config = {
        "app": "main:app",
        "host": settings.HOST,
        "port": settings.PORT,
        "reload": settings.DEBUG,
        "log_level": "info" if not settings.DEBUG else "debug",
        "access_log": True,
        "workers": 1,  # Single worker for local LLM to avoid memory issues
    }
    
    logger.info(f"🚀 Starting AutoLawyer on {settings.HOST}:{settings.PORT}")
    logger.info(f"📊 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🔧 Debug mode: {settings.DEBUG}")
    
    uvicorn.run(**config)