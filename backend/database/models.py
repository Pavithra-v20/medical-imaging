"""
SQLAlchemy ORM models for the Medical AI Diagnostic System.

Tables:
  - technicians: Medical staff who upload images and trigger diagnoses.
  - physicians: Doctors who review diagnostic reports.
  - patients: Patient demographic records.
  - diagnostic_sessions: Central table linking all agent outputs per session.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Date, Numeric,
    ForeignKey, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def gen_uuid():
    """Generate a new UUID4 string as default PK."""
    return str(uuid.uuid4())


class Technician(Base):
    """Medical imaging technician who operates the system."""

    __tablename__ = "technicians"

    technician_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(Text, nullable=False)
    role = Column(String(20), default="technician")   # 'technician' | 'admin'
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime, nullable=True)

    sessions = relationship("DiagnosticSession", back_populates="technician")


class Physician(Base):
    """Physician who reviews and annotates diagnostic reports."""

    __tablename__ = "physicians"

    physician_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name = Column(String(100), nullable=False)
    specialization = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)

    sessions = relationship("DiagnosticSession", back_populates="physician")


class Patient(Base):
    """Patient demographic and contact information."""

    __tablename__ = "patients"

    patient_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name = Column(String(100), nullable=False)
    dob = Column(Date, nullable=False)
    gender = Column(String(10), nullable=False)
    contact = Column(String(20), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    sessions = relationship("DiagnosticSession", back_populates="patient")


class DiagnosticSession(Base):
    """
    Central session record that aggregates all agent outputs for one diagnostic run.

    Lifecycle:
      status = 'pending'   → Master Agent starts
      status = 'complete'  → Agent 3 finishes writing the report link
      status = 'reviewed'  → Physician adds physician_review_notes
      status = 'error'     → Any agent failure; error details logged here
    """

    __tablename__ = "diagnostic_sessions"

    session_id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)

    technician_id = Column(UUID(as_uuid=False), ForeignKey("technicians.technician_id"), nullable=False)
    patient_id = Column(UUID(as_uuid=False), ForeignKey("patients.patient_id"), nullable=False)
    physician_id = Column(UUID(as_uuid=False), ForeignKey("physicians.physician_id"), nullable=False)

    # Master Agent
    image_path = Column(Text, nullable=True)

    # Agent 1 — Medical Image Agent
    modality = Column(String(20), nullable=True)               # 'mri' | 'xray' | 'ct'
    vista_output = Column(JSONB, nullable=True)                 # raw model probabilities
    diagnosis_output = Column(JSONB, nullable=True)            # full Gemini JSON
    confidence_score = Column(Numeric(5, 2), nullable=True)    # top class %
    clinical_notes = Column(Text, nullable=True)               # Gemini brief note
    model_used = Column(String(50), default="ViT-L16-fe + Xception")

    # Agent 2 — Visual Explanation Agent
    heatmap_path = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)

    # Agent 3 — Report Agent
    report_link = Column(Text, nullable=True)

    # Physician review (filled after report delivery)
    physician_review_notes = Column(Text, nullable=True)
    physician_reviewed_at = Column(DateTime, nullable=True)

    # Lifecycle
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, server_default=func.now())

    technician = relationship("Technician", back_populates="sessions")
    physician = relationship("Physician", back_populates="sessions")
    patient = relationship("Patient", back_populates="sessions")
