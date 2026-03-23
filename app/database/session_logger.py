import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.database.models import Base, Session as SessionModel
from app.utils.exceptions import DatabaseError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables on startup if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


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
