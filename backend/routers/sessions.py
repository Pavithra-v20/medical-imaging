"""
Router: Diagnostic Sessions
==============================
Endpoints for triggering the full AI diagnostic pipeline and retrieving results.

Routes:
  POST /sessions/run         — Upload image + submit session, runs all 3 agents
  GET  /sessions/            — List all sessions (paginated)
  GET  /sessions/{id}        — Get a specific session by ID
  PUT  /sessions/{id}/review — Physician adds review notes
  GET  /sessions/{id}/heatmap — Serve heatmap image file
"""
import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.database.models import DiagnosticSession, Patient, Technician, Physician
from backend.database.schemas import SessionRead, PhysicianReviewUpdate
from backend.agents.orchestrator import run_diagnostic_pipeline
from backend.agents.state import DiagnosticState
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("/run", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
async def run_session(
    patient_id:    str = Form(...),
    technician_id: str = Form(...),
    physician_id:  str = Form(...),
    modality:      str = Form("mri"),
    image:         UploadFile = File(...),
    db:            Session = Depends(get_db),
):
    """
    Trigger a new full diagnostic session.

    Uploads the medical image, saves it to disk, creates a pending session row,
    then runs the full LangGraph pipeline (Agent 1 → 2 → 3) synchronously.

    Args:
        patient_id:    UUID of an existing patient.
        technician_id: UUID of the logged-in technician.
        physician_id:  UUID of the assigned physician.
        modality:      Imaging modality ('mri', 'xray', 'ct').
        image:         Uploaded image file (JPEG, PNG).
        db:            Database session (injected).

    Returns:
        SessionRead: The completed diagnostic session record.

    Raises:
        404: If patient, technician, or physician ID is not found.
        422: If the image cannot be read.
        500: If the pipeline fails unexpectedly.
    """
    # ── Validate references ────────────────────────────────────────────────────
    for model_cls, uid, label in [
        (Patient,     patient_id,    "Patient"),
        (Technician,  technician_id, "Technician"),
        (Physician,   physician_id,  "Physician"),
    ]:
        if not db.query(model_cls).filter(
            getattr(model_cls, f"{label.lower()}_id") == uid
        ).first():
            raise HTTPException(status_code=404, detail=f"{label} '{uid}' not found.")

    # ── Save uploaded image ────────────────────────────────────────────────────
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    session_id  = str(uuid.uuid4())
    ext         = os.path.splitext(image.filename or "scan.jpg")[1].lower() or ".jpg"
    image_fname = f"{session_id}{ext}"
    image_path  = os.path.join(settings.UPLOAD_DIR, image_fname)

    image_bytes = await image.read()

    # DICOM conversion
    if ext in [".dcm", ".dicom"]:
        import pydicom
        import io
        from PIL import Image
        import numpy as np
        try:
            ds = pydicom.dcmread(io.BytesIO(image_bytes))
            pixel_array = ds.pixel_array
            
            # Normalize to 0-255 if needed
            if pixel_array.max() > 0:
                pixel_array = pixel_array - pixel_array.min()
                pixel_array = pixel_array / pixel_array.max()
                pixel_array = (pixel_array * 255).astype(np.uint8)
                
            img = Image.fromarray(pixel_array)
            out_io = io.BytesIO()
            img.save(out_io, format="JPEG")
            
            # We now treat it as a JPG downstream
            ext = ".jpg"
            image_fname = f"{session_id}{ext}"
            image_path  = os.path.join(settings.UPLOAD_DIR, image_fname)
            image_bytes = out_io.getvalue()
        except Exception as e:
            logger.error(f"Failed to process DICOM image: {e}")
            raise HTTPException(status_code=422, detail="Failed to read or convert DICOM file.")

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # ── Create pending DB session row ──────────────────────────────────────────
    session_row = DiagnosticSession(
        session_id    = session_id,
        technician_id = technician_id,
        patient_id    = patient_id,
        physician_id  = physician_id,
        image_path    = image_path,
        modality      = modality,
        status        = "pending",
    )
    db.add(session_row)
    db.commit()

    # ── Resolve patient metadata for Gemini prompt ─────────────────────────────
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()

    # ── Build initial LangGraph state ──────────────────────────────────────────
    initial_state: DiagnosticState = {
        "session_id":      session_id,
        "image_bytes":     image_bytes,
        "image_path":      image_path,
        "patient_id":      patient_id,
        "technician_id":   technician_id,
        "physician_id":    physician_id,
        "modality":        modality,
        "patient_name":    patient.name,
        "patient_dob":     str(patient.dob),
        "patient_gender":  patient.gender,
        "patient_contact": patient.contact or "",
        "status":          "pending",
    }

    # ── Run the pipeline ───────────────────────────────────────────────────────
    logger.info("Starting pipeline for session %s", session_id)
    final_state = run_diagnostic_pipeline(initial_state, db)

    # Refresh the row from DB so we return the latest data
    db.refresh(session_row)

    if final_state.get("status") == "error":
        raise HTTPException(
            status_code=500,
            detail=final_state.get("error", "Pipeline failed with unknown error"),
        )

    return session_row


@router.get("/", response_model=List[SessionRead])
def list_sessions(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """
    List all diagnostic sessions, newest first.

    Args:
        skip:  Number of records to skip (pagination offset).
        limit: Maximum number of records to return.
        db:    Database session.

    Returns:
        List[SessionRead]: Paginated session records.
    """
    rows = (
        db.query(DiagnosticSession)
        .order_by(DiagnosticSession.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return rows


@router.get("/{session_id}", response_model=SessionRead)
def get_session(session_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a single diagnostic session by its UUID.

    Args:
        session_id: UUID string of the session.
        db:         Database session.

    Returns:
        SessionRead: Full session record.

    Raises:
        404: If session ID is not found.
    """
    row = db.query(DiagnosticSession).filter(
        DiagnosticSession.session_id == session_id
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return row


@router.put("/{session_id}/review", response_model=SessionRead)
def physician_review(
    session_id: str,
    body: PhysicianReviewUpdate,
    db: Session = Depends(get_db),
):
    """
    Add physician review notes to a completed session.

    Sets `physician_review_notes`, `physician_reviewed_at`, and status to 'reviewed'.

    Args:
        session_id: UUID of the session being reviewed.
        body:       PhysicianReviewUpdate with review notes.
        db:         Database session.

    Returns:
        SessionRead: Updated session record.

    Raises:
        404: If session is not found.
        400: If session is not in 'complete' status.
    """
    row = db.query(DiagnosticSession).filter(
        DiagnosticSession.session_id == session_id
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if row.status not in ("complete", "reviewed"):
        raise HTTPException(status_code=400,
                            detail="Session must be 'complete' before review.")

    row.physician_review_notes = body.physician_review_notes
    row.physician_reviewed_at  = datetime.utcnow()
    row.status                 = "reviewed"
    db.commit()
    db.refresh(row)
    return row


@router.get("/{session_id}/heatmap")
def get_heatmap(session_id: str, db: Session = Depends(get_db)):
    """
    Serve the Grad-CAM heatmap PNG for a completed session.

    Args:
        session_id: UUID of the session.
        db:         Database session.

    Returns:
        FileResponse: PNG image file.

    Raises:
        404: If session or heatmap file is not found.
    """
    row = db.query(DiagnosticSession).filter(
        DiagnosticSession.session_id == session_id
    ).first()
    if not row or not row.heatmap_path:
        raise HTTPException(status_code=404, detail="Heatmap not found.")
    if not os.path.isfile(row.heatmap_path):
        raise HTTPException(status_code=404, detail="Heatmap file missing from disk.")
    return FileResponse(row.heatmap_path, media_type="image/png")


@router.get("/{session_id}/scan")
def get_scan(session_id: str, db: Session = Depends(get_db)):
    """
    Serve the original uploaded scan image for a session.

    Args:
        session_id: UUID of the session.
        db:         Database session.

    Returns:
        FileResponse: Image file (JPEG or PNG).

    Raises:
        404: If session or image file is not found.
    """
    row = db.query(DiagnosticSession).filter(
        DiagnosticSession.session_id == session_id
    ).first()
    if not row or not row.image_path:
        raise HTTPException(status_code=404, detail="Image not found.")
    if not os.path.isfile(row.image_path):
        raise HTTPException(status_code=404, detail="Image file missing from disk.")
    return FileResponse(row.image_path)
