"""
Router: Patients CRUD
======================
Routes:
  POST /patients/      — Register a new patient
  GET  /patients/      — List all patients
  GET  /patients/{id}  — Get patient by ID
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import Patient
from backend.database.schemas import PatientCreate, PatientRead

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.post("/", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def create_patient(body: PatientCreate, db: Session = Depends(get_db)):
    """
    Register a new patient in the system.

    Args:
        body: PatientCreate schema with name, dob, gender, contact.
        db:   Database session.

    Returns:
        PatientRead: Newly created patient record.
    """
    patient = Patient(**body.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("/", response_model=List[PatientRead])
def list_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve a paginated list of all patients.

    Args:
        skip:  Pagination offset.
        limit: Max records returned.
        db:    Database session.

    Returns:
        List[PatientRead]: Patient records.
    """
    return db.query(Patient).offset(skip).limit(limit).all()


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a single patient by UUID.

    Args:
        patient_id: UUID string of the patient.
        db:         Database session.

    Returns:
        PatientRead: Patient record.

    Raises:
        404: If patient not found.
    """
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return patient
