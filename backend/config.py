import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    # SQLite for local development, can be extended for PostgreSQL
    database_url: str = "sqlite:///./autolawyer.db"
    echo_sql: bool = False  # Set to True for SQL query logging
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Database initialization
    create_tables: bool = True
    migrate_on_startup: bool = True

@dataclass
class LLMConfig:
    """LLM and AI configuration"""
    # Model configuration
    model_config_path: str = "backend/local_llm/model_config.json"
    models_directory: str = "models"
    cache_directory: str = "backend/local_llm/cache"
    
    # Performance settings
    preload_model_on_startup: bool = True
    max_concurrent_requests: int = 3
    request_timeout_seconds: int = 300
    
    # Safety and limits
    max_tokens_per_request: int = 4096
    max_requests_per_minute: int = 60
    enable_content_filtering: bool = True
    
    # Agent settings
    max_agent_retries: int = 3
    agent_timeout_seconds: int = 180

@dataclass
class ServerConfig:
    """Server and API configuration"""
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    reload: bool = False  # Auto-reload on code changes
    
    # API settings
    api_title: str = "AutoLawyer API"
    api_description: str = "Local LLM-powered Legal Document Assistant"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # CORS settings
    allow_origins: list = field(default_factory=lambda: ["http://localhost:3000"])
    allow_credentials: bool = True
    allow_methods: list = field(default_factory=lambda: ["*"])
    allow_headers: list = field(default_factory=lambda: ["*"])
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30

@dataclass
class StorageConfig:
    """File storage and document management"""
    # Document storage
    documents_directory: str = "storage/documents"
    uploads_directory: str = "storage/uploads"
    exports_directory: str = "storage/exports"
    
    # File limits
    max_file_size_mb: int = 50
    allowed_file_types: list = field(default_factory=lambda: [
        ".pdf", ".docx", ".txt", ".md", ".rtf"
    ])
    
    # Document retention
    auto_cleanup_days: int = 30
    backup_documents: bool = True

@dataclass
class LegalConfig:
    """Legal-specific configuration"""
    # Citation and formatting
    default_citation_style: str = "bluebook"
    default_jurisdiction: str = "federal"
    
    # Document templates
    templates_directory: str = "templates/legal"
    
    # Compliance and disclaimers
    require_disclaimers: bool = True
    confidentiality_notices: bool = True
    ethical_guidelines_check: bool = True
    
    # Agent behavior
    max_research_depth: int = 5
    require_source_citations: bool = True
    fact_checking_enabled: bool = True

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    # Log levels
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Log files
    log_directory: str = "logs"
    log_file: str = "autolawyer.log"
    max_log_file_size_mb: int = 10
    log_backup_count: int = 5
    
    # Performance logging
    log_llm_performance: bool = True
    log_agent_interactions: bool = True
    log_user_actions: bool = False  # Privacy consideration
    
    # Monitoring
    enable_metrics: bool = True
    metrics_endpoint: str = "/metrics"

@dataclass
class AgentConfig:
    """Multi-agent system configuration"""
    # Available agents
    available_agents: list = field(default_factory=lambda: [
        "researcher", "drafter", "summarizer"
    ])
    
    # Agent workflow
    max_workflow_steps: int = 10
    enable_agent_collaboration: bool = True
    parallel_execution: bool = False  # Sequential by default for legal accuracy
    
    # Quality control
    enable_peer_review: bool = True
    require_consensus: bool = False
    confidence_threshold: float = 0.7

class Config:
    """Main application configuration class"""
    
    def __init__(self, env: Optional[Environment] = None):
        """Initialize configuration based on environment"""
        
        # Determine environment
        self.environment = env or Environment(
            os.getenv("AUTOLAWYER_ENV", Environment.DEVELOPMENT.value)
        )
        
        # Create directories
        self._create_directories()
        
        # Initialize configuration sections
        self.database = DatabaseConfig()
        self.llm = LLMConfig()
        self.server = ServerConfig()
        self.storage = StorageConfig()
        self.legal = LegalConfig()
        self.logging = LoggingConfig()
        self.agents = AgentConfig()
        
        # Apply environment-specific overrides
        self._apply_environment_config()
        
        # Load from environment variables
        self._load_from_env()
        
        logger.info(f"Configuration loaded for {self.environment.value} environment")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            "backend/local_llm/cache",
            "storage/documents",
            "storage/uploads", 
            "storage/exports",
            "logs",
            "models",
            "templates/legal"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def _apply_environment_config(self):
        """Apply environment-specific configuration overrides"""
        
        if self.environment == Environment.DEVELOPMENT:
            self.server.debug = True
            self.server.reload = True
            self.database.echo_sql = True
            self.logging.log_level = LogLevel.DEBUG
            self.llm.preload_model_on_startup = False  # Faster startup in dev
            
        elif self.environment == Environment.PRODUCTION:
            self.server.debug = False
            self.server.reload = False
            self.database.echo_sql = False
            self.logging.log_level = LogLevel.INFO
            self.llm.preload_model_on_startup = True
            self.server.secret_key = os.getenv("SECRET_KEY", self.server.secret_key)
            
        elif self.environment == Environment.TESTING:
            self.database.database_url = "sqlite:///./test_autolawyer.db"
            self.logging.log_level = LogLevel.WARNING
            self.llm.preload_model_on_startup = False
            self.llm.max_concurrent_requests = 1
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # Database
        if db_url := os.getenv("DATABASE_URL"):
            self.database.database_url = db_url
        
        # Server
        self.server.host = os.getenv("HOST", self.server.host)
        self.server.port = int(os.getenv("PORT", self.server.port))
        
        # LLM
        if model_config := os.getenv("MODEL_CONFIG_PATH"):
            self.llm.model_config_path = model_config
        
        if models_dir := os.getenv("MODELS_DIRECTORY"):
            self.llm.models_directory = models_dir
        
        # Security
        if secret_key := os.getenv("SECRET_KEY"):
            self.server.secret_key = secret_key
        
        # Logging
        if log_level := os.getenv("LOG_LEVEL"):
            try:
                self.logging.log_level = LogLevel(log_level.upper())
            except ValueError:
                logger.warning(f"Invalid log level: {log_level}, using default")
    
    def get_database_url(self) -> str:
        """Get the database URL for SQLAlchemy"""
        return self.database.database_url
    
    def get_llm_config_path(self) -> str:
        """Get the LLM configuration file path"""
        return self.llm.model_config_path
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for debugging/logging)"""
        return {
            "environment": self.environment.value,
            "database": {
                "database_url": self.database.database_url.replace(
                    self.database.database_url.split("://")[1].split("@")[0] + "@", "***@"
                ) if "@" in self.database.database_url else self.database.database_url,
                "echo_sql": self.database.echo_sql
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "debug": self.server.debug
            },
            "llm": {
                "model_config_path": self.llm.model_config_path,
                "preload_model": self.llm.preload_model_on_startup,
                "max_concurrent_requests": self.llm.max_concurrent_requests
            },
            "logging": {
                "level": self.logging.log_level.value,
                "directory": self.logging.log_directory
            }
        }

# Global configuration instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get or create the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config

def reload_config(env: Optional[Environment] = None) -> Config:
    """Reload the global configuration"""
    global _config
    _config = Config(env)
    return _config

# Environment variable helpers
def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key, "").lower()
    return value in ("true", "1", "yes", "on") if value else default

def get_env_int(key: str, default: int = 0) -> int:
    """Get integer value from environment variable"""
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

def get_env_list(key: str, default: list = None, separator: str = ",") -> list:
    """Get list value from environment variable"""
    if default is None:
        default = []
    
    value = os.getenv(key)
    if not value:
        return default
    
    return [item.strip() for item in value.split(separator) if item.strip()]