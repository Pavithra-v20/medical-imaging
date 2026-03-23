"""
Router: Physicians CRUD
========================
Routes:
  POST /physicians/      — Add a new physician
  GET  /physicians/      — List all physicians
  GET  /physicians/{id}  — Get physician by ID
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import Physician
from backend.database.schemas import PhysicianCreate, PhysicianRead

router = APIRouter(prefix="/physicians", tags=["Physicians"])


@router.post("/", response_model=PhysicianRead, status_code=status.HTTP_201_CREATED)
def create_physician(body: PhysicianCreate, db: Session = Depends(get_db)):
    """
    Add a new physician to the system.

    Args:
        body: PhysicianCreate schema with name, specialization, email.
        db:   Database session.

    Returns:
        PhysicianRead: Newly created physician record.

    Raises:
        400: If email already exists.
    """
    if db.query(Physician).filter(Physician.email == body.email).first():
        raise HTTPException(status_code=400, detail="Physician email already registered.")
    physician = Physician(**body.model_dump())
    db.add(physician)
    db.commit()
    db.refresh(physician)
    return physician


@router.get("/", response_model=List[PhysicianRead])
def list_physicians(db: Session = Depends(get_db)):
    """
    List all physicians.

    Args:
        db: Database session.

    Returns:
        List[PhysicianRead]: All physician records.
    """
    return db.query(Physician).all()


@router.get("/{physician_id}", response_model=PhysicianRead)
def get_physician(physician_id: str, db: Session = Depends(get_db)):
    """
    Get a single physician by UUID.

    Args:
        physician_id: UUID string of the physician.
        db:           Database session.

    Returns:
        PhysicianRead: Physician record.

    Raises:
        404: If physician not found.
    """
    rec = db.query(Physician).filter(Physician.physician_id == physician_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail=f"Physician '{physician_id}' not found.")
    return rec
