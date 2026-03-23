import numpy as np
from app.config import get_settings
from app.prediction.ct_chat_client import call_ct_chat
from app.prediction.medgemma_client import call_medgemma
from app.prediction.cnn_fallback import call_cnn_fallback
from app.utils.exceptions import PredictionError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def predict_disease(mask_metrics: dict, volume: np.ndarray) -> dict:
    """
    Route the prediction request to the appropriate model.
    - If PREDICTION_MODEL is 'ctchat', it is called directly.
    - If PREDICTION_MODEL is 'medgemma', it attempts to call MedGemma.
      If that fails, it falls back to a local CNN model.
    """
    # Build a concise payload for LLMs
    prediction_payload = {
        "lesion_count": mask_metrics["lesion_count"],
        "max_lesion_size_mm": mask_metrics["max_lesion_size_mm"],
        "has_critical_lesion": mask_metrics["has_critical_lesion"],
        "lesion_findings": [
            {
                "structure_class": f["structure_class"],
                "size_mm": f["size_mm"],
            }
            for f in mask_metrics.get("lesion_findings", [])
        ],
    }

    model = settings.prediction_model.lower()
    logger.info("prediction_started", model=model, lesion_count=mask_metrics["lesion_count"])

    if not settings.prediction_enabled:
        logger.info("prediction_disabled")
        result = {
            "disease_label": "Unknown",
            "confidence": 0.0,
            "reasoning": "Prediction disabled for local development.",
            "model_used": "disabled",
            "raw_response": "",
        }
    elif model == "local_rules":
        lesion_count = mask_metrics.get("lesion_count", 0)
        max_size = mask_metrics.get("max_lesion_size_mm", 0.0)
        organ_count = mask_metrics.get("organ_count", 0)
        if lesion_count > 0 or max_size >= 6.0:
            result = {
                "disease_label": "Abnormality Detected",
                "confidence": 0.75 if max_size >= 20.0 else 0.65,
                "reasoning": "Local heuristic: lesion presence/size above threshold.",
                "model_used": "local_rules",
                "raw_response": "",
            }
        else:
            result = {
                "disease_label": "No Abnormality Detected",
                "confidence": 0.6 if organ_count > 0 else 0.5,
                "reasoning": "Local heuristic: no lesions found in segmentation.",
                "model_used": "local_rules",
                "raw_response": "",
            }
    elif model == "ctchat":
        result = await call_ct_chat(prediction_payload)
    elif model == "medgemma":
        try:
            result = await call_medgemma(prediction_payload, volume)
        except PredictionError as e:
            logger.warning("medgemma_call_failed", error=str(e))
            # Fallback to local CNN model
            result = await call_cnn_fallback(volume)
    else:
        raise PredictionError(
            f"Unknown PREDICTION_MODEL '{model}'. Must be 'ctchat', 'medgemma', or 'local_rules'."
        )

    # Attach the full findings list to the result for the report builder
    result["findings"] = mask_metrics["findings"]

    return result
