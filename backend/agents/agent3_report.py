"""
Agent 3 — Report Agent
========================
The final agent in the pipeline. Takes all outputs from Agents 1 and 2 and
performs three sequential actions:

  1. PDF Generation:
     Calls `pdf_generator.build_report()` to produce a comprehensive
     ReportLab PDF containing: patient demographics, diagnosis table,
     original scan + heatmap side by side, clinical explanation, and a
     blank physician review section.

  2. Google Drive Upload:
     Uploads the PDF to the configured Drive folder via the service account
     and retrieves a publicly shareable link.

  3. Database Write:
     Writes a single updated row to `diagnostic_sessions` with all agent
     outputs and changes status to 'complete'.
"""
import logging

from sqlalchemy.orm import Session

from backend.agents.state import DiagnosticState
from backend.database.models import DiagnosticSession
from backend.reports.pdf_generator import build_report

logger = logging.getLogger(__name__)


def run_agent3(state: DiagnosticState, db: Session) -> DiagnosticState:
    """
    Execute Agent 3: PDF generation → Drive upload → DB write.

    Only runs if state["status"] == "agent2_complete".

    Reads:
        All Agent 1 and Agent 2 output fields from state.
        db: SQLAlchemy Session for writing the final session row.

    Writes:
        state["pdf_path"]    — local path to the generated PDF
        state["report_link"] — Google Drive shareable URL
        state["status"]      — 'complete'

    Side effects:
        - Saves a PDF file to REPORTS_DIR
        - Uploads the PDF to Google Drive
        - Updates the `diagnostic_sessions` row in PostgreSQL

    Args:
        state: Current DiagnosticState (must have Agent 2 outputs).
        db:    SQLAlchemy database session.

    Returns:
        Updated DiagnosticState with Agent 3 fields populated.
    """
    session_id = state.get("session_id", "unknown")
    logger.info("[Agent 3] Starting for session %s", session_id)

    if state.get("status") == "error":
        logger.warning("[Agent 3] Skipping — upstream error detected.")
        return state

    try:
        # ── Step 1: Generate PDF ───────────────────────────────────────────────
        logger.info("[Agent 3] Step 1 — Building PDF report …")
        pdf_path = build_report(state)
        state["pdf_path"] = pdf_path
        logger.info("[Agent 3] PDF saved to %s", pdf_path)

        # ── Step 2: Upload to Google Drive ─────────────────────────────────────
        # (Upload removed as per request)
        report_link = ""
        state["report_link"] = report_link

        # ── Step 3: Write to Database ──────────────────────────────────────────
        logger.info("[Agent 3] Step 3 — Writing session to database …")
        session_row = db.query(DiagnosticSession).filter(
            DiagnosticSession.session_id == session_id
        ).first()

        if session_row is None:
            logger.error("[Agent 3] Session row %s not found in DB!", session_id)
            raise ValueError(f"session_row {session_id} missing from DB")

        # Agent 1 fields
        session_row.modality        = state.get("modality", "mri")
        session_row.vista_output    = state.get("raw_probabilities")
        session_row.diagnosis_output = state.get("diagnosis_output")
        session_row.confidence_score = state.get("confidence_score")
        session_row.clinical_notes  = state.get("clinical_notes")
        session_row.model_used      = state.get("model_used", "ViT-L16-fe + Xception")

        # Agent 2 fields
        session_row.heatmap_path = state.get("heatmap_path")
        session_row.explanation  = state.get("explanation")

        # Agent 3 fields
        session_row.report_link = report_link
        session_row.status      = "complete"

        db.commit()
        db.refresh(session_row)
        logger.info("[Agent 3] DB write complete. Status = 'complete'.")

        state["status"] = "complete"

    except Exception as exc:
        logger.exception("[Agent 3] FAILED: %s", exc)
        db.rollback()

        # Update status to error in DB if possible
        try:
            session_row = db.query(DiagnosticSession).filter(
                DiagnosticSession.session_id == session_id
            ).first()
            if session_row:
                session_row.status = "error"
                db.commit()
        except Exception:
            pass

        state["status"] = "error"
        state["error"]  = f"Agent 3 failed: {exc}"

    return state
