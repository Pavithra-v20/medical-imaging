"""
SQLAlchemy database engine and session factory.

Provides:
  - `get_engine()` — lazy singleton engine (reads .env at first call)
  - `SessionLocal` — session factory bound to the lazy engine
  - `get_db`       — FastAPI dependency that yields a DB session
  - `init_db`      — auto-creates the database + all tables if missing
"""
import logging

from sqlalchemy import create_engine, text, engine_from_config
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import OperationalError
from typing import Generator

logger = logging.getLogger(__name__)

# ── Lazy engine ───────────────────────────────────────────────────────────────
# We intentionally delay engine creation until first use so that
# pydantic-settings / python-dotenv has time to load .env.

_engine = None


def _ensure_database_exists(db_url: str) -> None:
    """
    Connect to the 'postgres' maintenance DB and CREATE the target
    database if it does not already exist.

    Args:
        db_url: Full SQLAlchemy database URL (e.g. postgresql://user:pass@host:port/mydb)
    """
    if "/" not in db_url:
        logger.warning("Cannot parse DATABASE_URL to auto-create DB. Skipping.")
        return

    base_url = db_url.rsplit("/", 1)[0] + "/postgres"
    db_name  = db_url.rsplit("/", 1)[1].split("?")[0]  # strip query params

    try:
        tmp_engine = create_engine(
            base_url,
            isolation_level="AUTOCOMMIT",
            pool_pre_ping=True,
        )
        with tmp_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            ).fetchone()

            if not exists:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                logger.info("Database '%s' created automatically.", db_name)
            else:
                logger.info("Database '%s' already exists.", db_name)

        tmp_engine.dispose()

    except Exception as exc:
        logger.error("Could not auto-create database '%s': %s", db_name, exc)
        raise


def get_engine():
    """
    Return the singleton SQLAlchemy engine, creating it on first call.
    This defers engine creation until after settings have been loaded.
    """
    global _engine
    if _engine is None:
        # Import here so settings are loaded from .env first
        from backend.config import settings
        db_url = settings.DATABASE_URL

        _ensure_database_exists(db_url)

        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        logger.info("Database engine created for: %s", db_url.split("@")[-1])

    return _engine


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional DB session.

    Usage:
        @router.get("/")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Auto-create all tables (and the database itself) if they do not exist.

    - Connects to the 'postgres' maintenance DB to CREATE DATABASE if missing.
    - Then runs SQLAlchemy create_all() to create every model table.
    - Safe to call multiple times (idempotent — uses IF NOT EXISTS internally).

    Called once at FastAPI application startup via the lifespan handler.
    """
    from backend.database.models import Base

    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("All database tables are ready.")
    except OperationalError as exc:
        logger.error("Failed to create tables: %s", exc)
        raise
