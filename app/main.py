from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid

from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.file_utils import save_upload, cleanup_temp_files
from app.utils.exceptions import IngestionError, SegmentationError, PredictionError

from app.ingestion.dicom_loader import load_dicom
from app.ingestion.anonymizer import anonymize
from app.ingestion.converter import convert_to_nifti
from app.ingestion.preprocessor import preprocess

from app.segmentation.vista3d_client import run_segmentation
from app.segmentation.mask_parser import parse_masks

from app.prediction.predictor import predict_disease

from app.reporting.report_builder import build_report
from app.reporting.summarizer import summarize
from app.reporting.rev_req_rules import evaluate_rev_req

from app.database.session_logger import log_session
from app.database.review_queue import push_to_review_queue
from app.schemas.session_schema import SessionResponse

settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(
    title="CT Disease Agent",
    description="End-to-end CT scan disease prediction pipeline",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SessionResponse)
async def predict(
    file: UploadFile = File(..., description="DICOM CT scan (.dcm)"),
    technician_id: str = Form(...),
    patient_id: str = Form(...),
    physician_id: str = Form(...),
):
    session_id = str(uuid.uuid4())
    dicom_path = None
    nifti_path = None

    logger.info("session_started", session_id=session_id, technician_id=technician_id)

    try:
        # Step 1 — Save raw upload
        dicom_path = await save_upload(file, settings.storage_uploads, session_id)

        # Step 2 — Ingestion
        dicom_data = load_dicom(dicom_path)
        dicom_data = anonymize(dicom_data)
        nifti_path = convert_to_nifti(dicom_data, settings.storage_processed, session_id)
        nifti_volume = preprocess(nifti_path)

        # Step 3 — Segmentation
        raw_masks = await run_segmentation(nifti_volume, session_id)
        mask_metrics = parse_masks(raw_masks, nifti_volume)

        # Step 4 — Prediction
        prediction = await predict_disease(mask_metrics, nifti_volume)

        # Step 5 — Reporting
        report = build_report(prediction, mask_metrics, session_id)
        summary = summarize(prediction, mask_metrics)
        rev_req = evaluate_rev_req(prediction, mask_metrics)

        # Step 6 — Persist to DB
        await log_session(
            session_id=session_id,
            technician_id=technician_id,
            patient_id=patient_id,
            input_path=str(dicom_path),
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
        logger.error("segmentation_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=502, detail=f"Segmentation error: {e}")
    except PredictionError as e:
        logger.error("prediction_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=502, detail=f"Prediction error: {e}")
    except Exception as e:
        logger.error("unexpected_error", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        cleanup_temp_files([nifti_path])
