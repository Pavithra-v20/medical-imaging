from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid
from pathlib import Path

from app.config import get_settings
from app.logger_setup import setup_logging
from app.utils.logger import get_logger
from app.utils.file_utils import save_upload, cleanup_temp_files, extract_zip
from app.utils.exceptions import IngestionError, SegmentationError, PredictionError

from app.ingestion.dicom_loader import load_dicom, load_dicom_series, extract_metadata
from app.ingestion.anonymizer import anonymize, anonymize_series
from app.ingestion.converter import convert_to_nifti, convert_series_to_nifti, convert_dicom_dir_to_nifti
from app.ingestion.preprocessor import preprocess

from app.segmentation.vista3d_client import run_segmentation
from app.segmentation.lungmask_client import run_lungmask
from app.segmentation.totalseg_konfai_client import run_totalseg_konfai
from app.segmentation.mri_segmenter import run_mri_segmentation
import json
from app.segmentation.mask_parser import parse_masks, parse_lungmask_masks, parse_image_mask, parse_totalseg_masks
from app.segmentation.image_segmenter import segment_image

from app.prediction.predictor import predict_disease
from app.prediction.xray_classifier import classify_xray
from app.prediction.mr_classifier import classify_mr

from app.reporting.summarizer import summarize
from app.reporting.llm_explainer import generate_llm_explanation
from app.reporting.gemini_reporter import generate_gemini_report
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
    dicom_path = None
    dicom_dir = None
    zip_path = None
    nifti_path = None
    image_mask = None
    clinical_context = {}

    logger.info("session_started", session_id=session_id, technician_id=technician_id)

    try:
        # Step 1 — Save upload
        upload_path = await save_upload(file, settings.storage_uploads, session_id)

        # Step 2 — Ingestion (zip series, single DICOM, or image)
        file_ext = Path(file.filename).suffix.lower()
        is_image_input = file_ext in [".png", ".jpg", ".jpeg", ".webp"]
        if file_ext == ".zip":
            zip_path = upload_path
            dicom_dir = extract_zip(zip_path, Path(settings.storage_uploads) / session_id)
            slices = load_dicom_series(dicom_dir)
            if slices:
                clinical_context = extract_metadata(slices[0])
            slices = anonymize_series(slices)
            # Use SimpleITK conversion for better spacing/orientation
            nifti_path = convert_dicom_dir_to_nifti(dicom_dir, settings.storage_processed, session_id)
        elif file_ext == ".dcm":
            dicom_path = upload_path
            ds = load_dicom(dicom_path)
            clinical_context = extract_metadata(ds)
            ds = anonymize(ds)
            nifti_path = convert_to_nifti(ds, settings.storage_processed, session_id)
        elif is_image_input:
            logger.info("image_input_received", filename=file.filename)
        else:
            raise IngestionError(f"Unsupported file type: {file_ext}")

        # Step 3 — Segmentation (pass path, NIM resolves internally)
        parsed_prompts = json.loads(prompts) if prompts else None

        if settings.segmentation_enabled and not is_image_input:
            # We assume CT for now as per wiring instructions
            if settings.segmentation_backend.lower() == "vista3d":
                raw_masks = await run_segmentation(str(nifti_path), session_id, prompts=parsed_prompts)
                mask_metrics = parse_masks(raw_masks, None)
            elif settings.segmentation_backend.lower() == "lungmask":
                raw_masks, spacing_xyz_mm = await run_lungmask(str(nifti_path), session_id)
                mask_metrics = parse_lungmask_masks(raw_masks, spacing_xyz_mm)
            elif settings.segmentation_backend.lower() == "totalseg_konfai":
                raw_masks = run_totalseg_konfai(str(nifti_path), session_id)
                mask_metrics = parse_totalseg_masks(raw_masks)
            else:
                raise SegmentationError(
                    f"Unknown SEGMENTATION_BACKEND '{settings.segmentation_backend}'. "
                    "Use 'vista3d', 'lungmask', or 'totalseg_konfai'."
                )
        else:
            if is_image_input and settings.segmentation_enabled:
                # Lightweight 2D segmentation for image inputs
                mask, metrics = segment_image(str(upload_path))
                image_mask = mask
                mask_metrics = parse_image_mask(mask, metrics)
            else:
                if is_image_input:
                    logger.info("segmentation_skipped_for_image", session_id=session_id)
                else:
                    logger.info("segmentation_disabled", session_id=session_id)
                mask_metrics = {
                    "findings": [],
                    "lesion_findings": [],
                    "organ_count": 0,
                    "lesion_count": 0,
                    "max_lesion_size_mm": 0.0,
                    "has_critical_lesion": False,
                    "has_suspicious_lesion": False,
                }

        # Step 4 — Prediction
        if is_image_input and settings.prediction_model.lower() == "xray":
            prediction = await classify_xray(str(upload_path), image_mask)
        elif is_image_input and settings.prediction_model.lower() == "mr":
            prediction = await classify_mr(str(upload_path))
        else:
            prediction = await predict_disease(mask_metrics, None)

        # Step 5 — Summary (LLM-enhanced)
        explanation = generate_llm_explanation(prediction, mask_metrics, clinical_context)
        rev_req = evaluate_rev_req(prediction, mask_metrics)
        summary = explanation if explanation else summarize(prediction, mask_metrics, clinical_context, rev_req)

        # Step 6 — Report (Gemini, Markdown template)
        report = await generate_gemini_report(
            summary=summary,
            prediction=prediction,
            mask_metrics=mask_metrics,
            session_id=session_id,
            technician_id=technician_id,
            patient_id=patient_id,
            physician_id=physician_id,
            image_path=str(upload_path) if is_image_input else None,
            clinical_context=clinical_context,
        )
        rev_req = evaluate_rev_req(prediction, mask_metrics)

        # Step 7 — Persist to DB
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
            patient_age=clinical_context.get("patient_age"),
            patient_sex=clinical_context.get("patient_sex"),
            study_description=clinical_context.get("study_description"),
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
        cleanup_temp_files([zip_path, dicom_dir, nifti_path, dicom_path])
