"""
Router: Technicians — Registration & Auth
==========================================
Routes:
  POST /technicians/register  — Create new technician account
  POST /technicians/login     — JWT login
  GET  /technicians/          — List all technicians
  GET  /technicians/{id}      — Get technician by ID
"""
import logging
from datetime import datetime, timedelta
from typing import List

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.db import get_db
from backend.database.models import Technician
from backend.database.schemas import (
    TechnicianCreate, TechnicianRead, TechnicianLogin, TokenResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/technicians", tags=["Technicians"])
security = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _create_token(technician_id: str) -> str:
    """
    Create a JWT access token for a technician.

    Args:
        technician_id: UUID string to embed as the `sub` claim.

    Returns:
        str: Signed JWT token string.
    """
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": technician_id, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


@router.post("/register", response_model=TechnicianRead, status_code=status.HTTP_201_CREATED)
def register_technician(body: TechnicianCreate, db: Session = Depends(get_db)):
    """
    Register a new technician account.

    Args:
        body: TechnicianCreate with name, email, password, role.
        db:   Database session.

    Returns:
        TechnicianRead: Created technician (no password).

    Raises:
        400: If email already exists.
    """
    if db.query(Technician).filter(Technician.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")

    tech = Technician(
        name            = body.name,
        email           = body.email,
        hashed_password = _hash_password(body.password),
        role            = body.role,
    )
    db.add(tech)
    db.commit()
    db.refresh(tech)
    return tech


@router.post("/login", response_model=TokenResponse)
def login_technician(body: TechnicianLogin, db: Session = Depends(get_db)):
    """
    Authenticate a technician and return a JWT access token.

    Args:
        body: TechnicianLogin with email and password.
        db:   Database session.

    Returns:
        TokenResponse: JWT token + technician info.

    Raises:
        401: If credentials are invalid.
    """
    tech = db.query(Technician).filter(Technician.email == body.email).first()
    if not tech or not _verify_password(body.password, tech.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    tech.last_login = datetime.utcnow()
    db.commit()
    db.refresh(tech)

    token = _create_token(str(tech.technician_id))
    return TokenResponse(access_token=token, technician=TechnicianRead.from_orm(tech))


@router.get("/", response_model=List[TechnicianRead])
def list_technicians(db: Session = Depends(get_db)):
    """
    List all technicians.

    Args:
        db: Database session.

    Returns:
        List[TechnicianRead]: All technician records (no passwords).
    """
    return db.query(Technician).all()


@router.get("/{technician_id}", response_model=TechnicianRead)
def get_technician(technician_id: str, db: Session = Depends(get_db)):
    """
    Get a single technician by UUID.

    Args:
        technician_id: UUID string.
        db:            Database session.

    Returns:
        TechnicianRead: Technician record.

    Raises:
        404: If not found.
    """
    rec = db.query(Technician).filter(Technician.technician_id == technician_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail=f"Technician '{technician_id}' not found.")
    return rec
