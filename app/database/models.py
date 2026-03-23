from sqlalchemy import Column, String, Boolean, Text, DateTime, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone
import uuid


class Base(DeclarativeBase):
    pass


class Session(Base):
    """
    Maps to the 'sessions' table in PostgreSQL.
    One row per CT scan processed by the agent.

    Columns match your spec:
        session_id    — unique identifier for this run
        technician_id — who uploaded the scan
        patient_id    — anonymized patient reference
        input         — path to the original DICOM file
        output        — JSON string of prediction result
        summary       — short plain-language summary
        report        — full structured radiology report
        physician_id  — assigned reviewing physician
        rev_req       — True if physician review is required
    """
    __tablename__ = "sessions"

    session_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    technician_id = Column(String(64), nullable=False, index=True)
    patient_id = Column(String(64), nullable=False, index=True)
    input = Column(Text, nullable=False, comment="Path to original DICOM file")
    output = Column(Text, nullable=True, comment="JSON prediction result from AI model")
    summary = Column(Text, nullable=True, comment="Short plain-language summary")
    report = Column(Text, nullable=True, comment="Full structured radiology report")
    physician_id = Column(String(64), nullable=True, index=True)
    rev_req = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    confidence = Column(Float, nullable=True, comment="Model prediction confidence 0-1")
    patient_age = Column(String(16), nullable=True, comment="Extracted patient age")
    patient_sex = Column(String(16), nullable=True, comment="Extracted patient sex")
    study_description = Column(Text, nullable=True, comment="DICOM study description")

    def __repr__(self):
        return (
            f"<Session id={self.session_id} patient={self.patient_id} "
            f"rev_req={self.rev_req}>"
        )
