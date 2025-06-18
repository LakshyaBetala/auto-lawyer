"""
Database connection and session management for AutoLawyer
Handles SQLAlchemy setup, connection pooling, and session lifecycle
Compatible with agent dataclasses and local LLM architecture
"""

import os
import logging
from typing import Generator, Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration from environment or default to local SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./autolawyer.db")

# SQLAlchemy engine configuration
if DATABASE_URL.startswith("sqlite"):
    # SQLite configuration for local development
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False,  # Allow FastAPI threading
            "timeout": 30,  # 30 second timeout
            "isolation_level": None  # Autocommit mode
        },
        poolclass=StaticPool,
        pool_pre_ping=True,
        echo=False  # Set to True for SQL debugging
    )
else:
    # PostgreSQL/MySQL configuration for production
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections every hour
        echo=False
    )

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Keep objects accessible after commit
)

# Base class for all ORM models
Base = declarative_base()

# Metadata for schema management
metadata = MetaData()


def create_tables():
    """
    Create all database tables based on defined models
    Should be called during application startup
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        # Log table information
        inspector = create_engine(DATABASE_URL).connect()
        if DATABASE_URL.startswith("sqlite"):
            result = inspector.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            tables = [row[0] for row in result]
        else:
            result = inspector.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';"))
            tables = [row[0] for row in result]
        
        logger.info(f"📊 Tables created: {', '.join(tables)}")
        inspector.close()
        
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency function for database sessions
    Provides automatic session management with proper cleanup
    
    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db session here
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a database session for direct use outside of FastAPI
    Remember to close the session manually!
    
    Usage:
        db = get_db_session()
        try:
            # Use db session
            pass
        finally:
            db.close()
    """
    return SessionLocal()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions
    Automatically handles commit/rollback and cleanup
    
    Usage:
        with get_db_context() as db:
            # Use db session
            pass
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


class DatabaseManager:
    """
    High-level database manager for AutoLawyer operations
    Handles agent data storage, retrieval, and management
    """
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        self._connection_pool_size = 10
    
    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive database health check
        
        Returns:
            Dict with health status and metrics
        """
        try:
            with self.engine.connect() as conn:
                # Test basic connectivity
                conn.execute(text("SELECT 1"))
                
                # Get database info
                if DATABASE_URL.startswith("sqlite"):
                    # SQLite specific queries
                    size_result = conn.execute(text("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"))
                    db_size = size_result.fetchone()[0] if size_result else 0
                    
                    table_result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                    tables = [row[0] for row in table_result]
                else:
                    # PostgreSQL specific queries
                    size_result = conn.execute(text("SELECT pg_size_pretty(pg_database_size(current_database()))"))
                    db_size = size_result.fetchone()[0] if size_result else "Unknown"
                    
                    table_result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
                    tables = [row[0] for row in table_result]
                
                return {
                    "status": "healthy",
                    "database_url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
                    "database_size": db_size,
                    "tables": tables,
                    "table_count": len(tables),
                    "connection_pool_size": self._connection_pool_size,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def store_agent_data(self, agent_id: str, data_type: str, data: Dict[str, Any]) -> bool:
        """
        Store agent data (drafts, research, summaries, etc.)
        
        Args:
            agent_id: Agent identifier (drafter, researcher, summarizer)
            data_type: Type of data (draft, research_result, summary, etc.)
            data: Data dictionary to store
            
        Returns:
            bool: True if stored successfully
        """
        try:
            with get_db_context() as db:
                # Create generic storage record
                storage_record = {
                    "agent_id": agent_id,
                    "data_type": data_type,
                    "data": json.dumps(data),
                    "timestamp": datetime.now().isoformat(),
                    "data_id": data.get("id") or data.get("draft_id") or data.get("query_id")
                }
                
                # This would typically insert into an AgentData table
                # For now, we'll use raw SQL for flexibility
                if DATABASE_URL.startswith("sqlite"):
                    db.execute(text("""
                        INSERT OR REPLACE INTO agent_data 
                        (agent_id, data_type, data_id, data_json, timestamp)
                        VALUES (:agent_id, :data_type, :data_id, :data, :timestamp)
                    """), {
                        "agent_id": agent_id,
                        "data_type": data_type,
                        "data_id": storage_record["data_id"],
                        "data": storage_record["data"],
                        "timestamp": storage_record["timestamp"]
                    })
                
                logger.info(f"Stored {data_type} data for agent {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store agent data: {e}")
            return False
    
    def retrieve_agent_data(self, agent_id: str, data_type: str = None, data_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve agent data with optional filtering
        
        Args:
            agent_id: Agent identifier
            data_type: Optional data type filter
            data_id: Optional specific data ID
            
        Returns:
            List of data records
        """
        try:
            with get_db_context() as db:
                query = "SELECT * FROM agent_data WHERE agent_id = :agent_id"
                params = {"agent_id": agent_id}
                
                if data_type:
                    query += " AND data_type = :data_type"
                    params["data_type"] = data_type
                
                if data_id:
                    query += " AND data_id = :data_id"
                    params["data_id"] = data_id
                
                query += " ORDER BY timestamp DESC"
                
                result = db.execute(text(query), params)
                
                records = []
                for row in result:
                    try:
                        data = json.loads(row.data_json)
                        records.append({
                            "agent_id": row.agent_id,
                            "data_type": row.data_type,
                            "data_id": row.data_id,
                            "data": data,
                            "timestamp": row.timestamp
                        })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse data for record {row.data_id}: {e}")
                
                return records
                
        except Exception as e:
            logger.error(f"Failed to retrieve agent data: {e}")
            return []
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """
        Clean up old agent data to manage database size
        
        Args:
            days_old: Delete data older than this many days
            
        Returns:
            Number of records deleted
        """
        try:
            with get_db_context() as db:
                cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
                
                result = db.execute(text("""
                    DELETE FROM agent_data 
                    WHERE timestamp < :cutoff_date
                """), {"cutoff_date": cutoff_date.isoformat()})
                
                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} old records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about agent data storage
        
        Returns:
            Dictionary with agent statistics
        """
        try:
            with get_db_context() as db:
                # Get counts by agent and data type
                result = db.execute(text("""
                    SELECT agent_id, data_type, COUNT(*) as count
                    FROM agent_data
                    GROUP BY agent_id, data_type
                    ORDER BY agent_id, data_type
                """))
                
                stats = {}
                total_records = 0
                
                for row in result:
                    agent_id = row.agent_id
                    if agent_id not in stats:
                        stats[agent_id] = {}
                    
                    stats[agent_id][row.data_type] = row.count
                    total_records += row.count
                
                # Get latest activity
                latest_result = db.execute(text("""
                    SELECT agent_id, MAX(timestamp) as latest_activity
                    FROM agent_data
                    GROUP BY agent_id
                """))
                
                for row in latest_result:
                    if row.agent_id in stats:
                        stats[row.agent_id]["latest_activity"] = row.latest_activity
                
                return {
                    "total_records": total_records,
                    "agents": stats,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get agent statistics: {e}")
            return {"error": str(e)}
    
    def reset_database(self):
        """
        ⚠️ WARNING: Drops all tables and recreates them
        Only use in development!
        """
        try:
            logger.warning("🔄 Resetting database - ALL DATA WILL BE LOST!")
            Base.metadata.drop_all(bind=self.engine)
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database reset completed")
        except Exception as e:
            logger.error(f"❌ Error resetting database: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


def init_database():
    """
    Initialize the AutoLawyer database
    Creates tables and performs initial setup
    """
    logger.info("🚀 Initializing AutoLawyer database...")
    
    try:
        # Check database connection
        health = db_manager.health_check()
        if health["status"] != "healthy":
            raise Exception(f"Database connection failed: {health.get('error', 'Unknown error')}")
        
        # Create tables
        create_tables()
        
        # Create the basic agent_data table if it doesn't exist
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS agent_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data_id TEXT,
                    data_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    UNIQUE(agent_id, data_type, data_id)
                )
            """))
            
            # Create indexes for better performance
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_data_agent_id ON agent_data(agent_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_data_type ON agent_data(data_type)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_data_timestamp ON agent_data(timestamp)"))
            conn.commit()
        
        # Log final status
        final_health = db_manager.health_check()
        logger.info(f"✅ Database initialization complete!")
        logger.info(f"📊 Database info: {final_health}")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


# Utility functions for agent integration
def store_draft_result(draft_result) -> bool:
    """Store a DraftResult from the drafter agent"""
    return db_manager.store_agent_data(
        agent_id="drafter",
        data_type="draft_result",
        data=draft_result.to_dict()
    )


def store_research_result(research_result) -> bool:
    """Store a ResearchResult from the researcher agent"""
    return db_manager.store_agent_data(
        agent_id="researcher", 
        data_type="research_result",
        data=research_result.to_dict()
    )


def get_recent_drafts(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent draft results"""
    records = db_manager.retrieve_agent_data("drafter", "draft_result")
    return records[:limit]


def get_research_by_query(query_id: str) -> Optional[Dict[str, Any]]:
    """Get research result by query ID"""
    records = db_manager.retrieve_agent_data("researcher", "research_result", query_id)
    return records[0] if records else None


if __name__ == "__main__":
    # Test database setup when run directly
    print("🧪 Testing AutoLawyer database setup...")
    init_database()
    
    # Test health check
    health = db_manager.health_check()
    print(f"Health check: {health}")
    
    print("✅ Database test completed!")