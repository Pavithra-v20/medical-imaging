from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid
from pathlib import Path

from app.config import get_settings
from app.logger_setup import setup_logging
from app.utils.logger import get_logger
from app.utils.file_utils import save_upload, cleanup_temp_files
from app.utils.exceptions import IngestionError, PredictionError
from app.prediction.mr_classifier import classify_mr

from app.reporting.summarizer import summarize
from app.reporting.llm_explainer import generate_llm_explanation
from app.reporting.rev_req_rules import evaluate_rev_req

from app.database.session_logger import log_session, init_db
from app.database.review_queue import push_to_review_queue
from app.schemas.session_schema import SessionResponse

settings = get_settings()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup: Initialize logging and database.
    """
    setup_logging()
    if settings.database_enabled:
        await init_db()
    else:
        logger.info("database_disabled")
    yield

app = FastAPI(
    title="CT Disease Agent",
    description="End-to-end CT scan disease prediction pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SessionResponse)
async def predict(
    file: UploadFile = File(..., description="Zipped DICOM series (.zip)"),
    technician_id: str = Form(...),
    patient_id: str = Form(...),
    physician_id: str = Form(...),
    prompts: str = Form(None, description="Optional JSON string for prompts, e.g. {'classes': ['lung']}"),
):
    session_id = str(uuid.uuid4())
    clinical_context = {}

    logger.info("session_started", session_id=session_id, technician_id=technician_id)

    try:
        # Step 1 ? Save upload
        upload_path = await save_upload(file, settings.storage_uploads, session_id)

        # Step 2 ? Validate input (images only)
        file_ext = Path(file.filename).suffix.lower()
        is_image_input = file_ext in [".png", ".jpg", ".jpeg", ".webp"]
        if not is_image_input:
            raise IngestionError("Only image files (png/jpg/jpeg/webp) are supported.")

        # Step 3 ? Prediction (MR model)
        prediction = await classify_mr(str(upload_path))

        # Step 4 ? Summary (LLM-enhanced)
        mask_metrics = {
            "findings": [],
            "lesion_findings": [],
            "organ_count": 0,
            "lesion_count": 0,
            "max_lesion_size_mm": 0.0,
            "has_critical_lesion": False,
            "has_suspicious_lesion": False,
        }
        explanation = generate_llm_explanation(prediction, mask_metrics)
        summary = explanation if explanation else summarize(prediction, mask_metrics)
        report = None
        rev_req = evaluate_rev_req(prediction, mask_metrics)

        # Step 5 ? Persist to DB
        await log_session(
            session_id=session_id,
            technician_id=technician_id,
            patient_id=patient_id,
            input_path=str(upload_path),
            output=prediction,
            summary=summary,
            report=report,
            physician_id=physician_id,
            rev_req=rev_req,
        )

        if rev_req:
            await push_to_review_queue(session_id, physician_id, prediction)
            logger.info("review_flagged", session_id=session_id)

        logger.info("session_complete", session_id=session_id, rev_req=rev_req)

        return SessionResponse(
            session_id=session_id,
            summary=summary,
            report=report,
            rev_req=rev_req,
            prediction=prediction,
        )
    except IngestionError as e:
        logger.error("ingestion_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=422, detail=f"Ingestion error: {e}")
    except SegmentationError as e:
        logger.error("segmentation_failed", session_id=session_id, error=str(e), detail=repr(e))
        raise HTTPException(status_code=502, detail=f"Segmentation error: {e}")
    except PredictionError as e:
        logger.error("prediction_failed", session_id=session_id, error=str(e), detail=repr(e))
        raise HTTPException(status_code=502, detail=f"Prediction error: {e}")
    except Exception as e:
        import traceback
        logger.error("unexpected_error", session_id=session_id, error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        cleanup_temp_files([upload_path])
