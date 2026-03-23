import json
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.database.models import Base, Session as SessionModel
from app.utils.exceptions import DatabaseError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

if settings.database_enabled:
    if not settings.database_url:
        raise ValueError("DATABASE_URL must be set when DATABASE_ENABLED=true")
    # We need a sync URL for sqlalchemy-utils and initial table creation
    sync_url = settings.database_url.replace("asyncpg", "psycopg2")
    engine_sync = create_engine(sync_url)
    engine = create_async_engine(settings.database_url, echo=False, future=True)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
else:
    sync_url = None
    engine_sync = None
    engine = None
    AsyncSessionLocal = None


async def init_db():
    """
    Ensure the database exists and create all tables.
    Uses a sync connection for DB/Table creation as it is more reliable for 
    DDL operations in Postgres.
    """
    if not settings.database_enabled:
        logger.info("database_disabled")
        return
    try:
        if not database_exists(sync_url):
            create_database(sync_url)
            logger.info("database_created", url=sync_url)
        
        # Create tables if they don't exist
        Base.metadata.create_all(engine_sync)
        logger.info("database_tables_initialized")
    except Exception as e:
        logger.error("database_init_failed", error=str(e))
        # We don't raise here to allow the app to attempt starting, 
        # but subsequent DB calls will fail.


async def log_session(
    session_id: str,
    technician_id: str,
    patient_id: str,
    input_path: str,
    output: dict,
    summary: str,
    report: str,
    physician_id: str,
    rev_req: bool,
) -> None:
    """
    Write one complete session record to the PostgreSQL sessions table.
    The 'output' dict is serialized to JSON text for storage.
    Raises DatabaseError on failure so the caller can handle it.
    """
    if not settings.database_enabled:
        logger.info("database_disabled_skip_log", session_id=session_id)
        return
    record = SessionModel(
        session_id=session_id,
        technician_id=technician_id,
        patient_id=patient_id,
        input=input_path,
        output=json.dumps(output),
        summary=summary,
        report=report,
        physician_id=physician_id,
        rev_req=rev_req,
        confidence=output.get("confidence"),
    )

    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                db.add(record)
        logger.info("session_logged", session_id=session_id, rev_req=rev_req)
    except Exception as e:
        raise DatabaseError(f"Failed to write session {session_id}: {e}")
