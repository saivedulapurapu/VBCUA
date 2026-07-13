import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.config import settings

logger = logging.getLogger(__name__)

# Attempt to connect to PostgreSQL. If it fails, fallback to SQLite.
db_url = settings.DATABASE_URL
engine = None

if db_url.startswith("postgresql"):
    try:
        # Create engine and try to connect
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20
        )
        # Test connection
        conn = engine.connect()
        conn.close()
        logger.info("Successfully connected to PostgreSQL database.")
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed: {e}. Falling back to SQLite local database.")
        db_url = "sqlite:///./vbcua.db"
        engine = None

if engine is None:
    # Connect to SQLite
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {}
    )
    logger.info(f"Using database: {db_url}")

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base
Base = declarative_base()

# Dependency to get db session in FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
