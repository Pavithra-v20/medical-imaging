"""
Pydantic schemas for request/response validation across all routers.

Follows a Base → Create → Read pattern for each model:
  - Base:   shared fields
  - Create: fields accepted on POST (write-only fields like password go here)
  - Read:   fields returned to client (excludes hashed_password, etc.)
"""
from pydantic import BaseModel, EmailStr
from typing import Optional, Any, Dict
from datetime import date, datetime
from decimal import Decimal


# ─────────────────────────────────────────────────────────────────────────────
# Technician
# ─────────────────────────────────────────────────────────────────────────────

class TechnicianCreate(BaseModel):
    """Schema for registering a new technician."""
    name: str
    email: EmailStr
    password: str
    role: str = "technician"


class TechnicianRead(BaseModel):
    """Public-safe technician view (no password)."""
    technician_id: str
    name: str
    email: str
    role: str
    created_at: Optional[datetime]
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class TechnicianLogin(BaseModel):
    """Login credentials for technician authentication."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token returned after successful login."""
    access_token: str
    token_type: str = "bearer"
    technician: TechnicianRead


# ─────────────────────────────────────────────────────────────────────────────
# Physician
# ─────────────────────────────────────────────────────────────────────────────

class PhysicianCreate(BaseModel):
    """Schema for adding a new physician."""
    name: str
    specialization: str
    email: EmailStr


class PhysicianRead(BaseModel):
    """Physician record returned to client."""
    physician_id: str
    name: str
    specialization: str
    email: str

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# Patient
# ─────────────────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    """Schema for registering a new patient."""
    name: str
    dob: date
    gender: str
    contact: Optional[str] = None


class PatientRead(BaseModel):
    """Patient record returned to client."""
    patient_id: str
    name: str
    dob: date
    gender: str
    contact: Optional[str]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Session
# ─────────────────────────────────────────────────────────────────────────────

class SessionRunRequest(BaseModel):
    """Request body to trigger a new diagnostic session via the orchestrator."""
    patient_id: str
    technician_id: str
    physician_id: str
    modality: str = "mri"       # default for brain tumour model


class PhysicianReviewUpdate(BaseModel):
    """Physician adds review notes to a completed session."""
    physician_review_notes: str


class SessionRead(BaseModel):
    """Full session record returned after pipeline completion."""
    session_id: str
    patient_id: str
    technician_id: str
    physician_id: str
    image_path: Optional[str]
    modality: Optional[str]
    vista_output: Optional[Dict[str, Any]]
    diagnosis_output: Optional[Dict[str, Any]]
    confidence_score: Optional[Decimal]
    clinical_notes: Optional[str]
    model_used: Optional[str]
    heatmap_path: Optional[str]
    explanation: Optional[str]
    report_link: Optional[str]
    physician_review_notes: Optional[str]
    physician_reviewed_at: Optional[datetime]
    status: str
    created_at: Optional[datetime]

    class Config:
        from_attributes = True
        protected_namespaces = ()
