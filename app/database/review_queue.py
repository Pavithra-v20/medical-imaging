from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Table, Column, String, DateTime, Text, MetaData
from datetime import datetime, timezone

from app.database.session_logger import AsyncSessionLocal, engine
from app.utils.exceptions import DatabaseError
from app.utils.logger import get_logger
import json

logger = get_logger(__name__)

metadata = MetaData()

review_queue_table = Table(
    "review_queue",
    metadata,
    Column("queue_id", String(64), primary_key=True),
    Column("session_id", String(64), nullable=False, index=True),
    Column("physician_id", String(64), nullable=False, index=True),
    Column("disease_label", String(256), nullable=True),
    Column("confidence", String(16), nullable=True),
    Column("status", String(32), nullable=False, default="pending"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("notes", Text, nullable=True),
)


async def init_review_queue():
    """Create review_queue table if it does not exist."""
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


async def push_to_review_queue(
    session_id: str,
    physician_id: str,
    prediction: dict,
) -> None:
    """
    Insert a flagged session into the review_queue table so the assigned
    physician can see it in their worklist.

    Status starts as 'pending' and is updated to 'reviewed' by the
    physician portal when they sign off.
    """
    import uuid
    queue_id = str(uuid.uuid4())

    values = {
        "queue_id": queue_id,
        "session_id": session_id,
        "physician_id": physician_id,
        "disease_label": prediction.get("disease_label", ""),
        "confidence": str(round(prediction.get("confidence", 0.0), 4)),
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "notes": None,
    }

    try:
        async with engine.begin() as conn:
            await conn.execute(review_queue_table.insert().values(**values))
        logger.info(
            "review_queued",
            queue_id=queue_id,
            session_id=session_id,
            physician_id=physician_id,
        )
    except Exception as e:
        raise DatabaseError(f"Failed to push session {session_id} to review queue: {e}")
