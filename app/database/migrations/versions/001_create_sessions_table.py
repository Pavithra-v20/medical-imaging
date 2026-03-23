"""create sessions and review_queue tables

Revision ID: 001
Create Date: 2025-01-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "sessions",
        sa.Column("session_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("technician_id", sa.String(64), nullable=False),
        sa.Column("patient_id", sa.String(64), nullable=False),
        sa.Column("input", sa.Text, nullable=False),
        sa.Column("output", sa.Text, nullable=True),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("report", sa.Text, nullable=True),
        sa.Column("physician_id", sa.String(64), nullable=True),
        sa.Column("rev_req", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index("ix_sessions_technician_id", "sessions", ["technician_id"])
    op.create_index("ix_sessions_patient_id", "sessions", ["patient_id"])
    op.create_index("ix_sessions_physician_id", "sessions", ["physician_id"])
    op.create_index("ix_sessions_rev_req", "sessions", ["rev_req"])

    op.create_table(
        "review_queue",
        sa.Column("queue_id", sa.String(64), primary_key=True),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("physician_id", sa.String(64), nullable=False),
        sa.Column("disease_label", sa.String(256), nullable=True),
        sa.Column("confidence", sa.String(16), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("notes", sa.Text, nullable=True),
    )

    op.create_index("ix_review_queue_session_id", "review_queue", ["session_id"])
    op.create_index("ix_review_queue_physician_id", "review_queue", ["physician_id"])


def downgrade():
    op.drop_table("review_queue")
    op.drop_table("sessions")
