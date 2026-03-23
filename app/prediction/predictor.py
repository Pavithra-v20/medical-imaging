import numpy as np
from app.config import get_settings
from app.prediction.ct_chat_client import call_ct_chat
from app.prediction.medgemma_client import call_medgemma
from app.utils.exceptions import PredictionError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def predict_disease(mask_metrics: dict, volume: np.ndarray) -> dict:
    """
    Route the prediction request to either CT-CHAT or MedGemma based on
    the PREDICTION_MODEL setting in .env.

    Packages mask_metrics into a trimmed payload (only lesion findings
    and summary stats) to keep the prompt concise, then calls the
    appropriate model client.

    Returns a standardized prediction dict:
        {
            disease_label: str,
            confidence: float,
            reasoning: str,
            model_used: str,
            findings: list,
            raw_response: str,
        }
    """
    # Build a concise payload — send only lesion findings + summary stats
    prediction_payload = {
        "lesion_count": mask_metrics["lesion_count"],
        "max_lesion_size_mm": mask_metrics["max_lesion_size_mm"],
        "has_critical_lesion": mask_metrics["has_critical_lesion"],
        "lesion_findings": [
            {
                "structure_class": f["structure_class"],
                "size_mm": f["size_mm"],
                "volume_cm3": f["volume_cm3"],
                "num_instances": f["num_instances"],
                "location_voxel": f["location_voxel"],
            }
            for f in mask_metrics["lesion_findings"]
        ],
        "organ_summary": [
            {
                "structure_class": f["structure_class"],
                "volume_cm3": f["volume_cm3"],
            }
            for f in mask_metrics["findings"]
            if not f["is_lesion"]
        ],
    }

    model = settings.prediction_model.lower()

    logger.info("prediction_started", model=model, lesion_count=mask_metrics["lesion_count"])

    if model == "ctchat":
        result = await call_ct_chat(prediction_payload)
    elif model == "medgemma":
        result = await call_medgemma(prediction_payload)
    else:
        raise PredictionError(
            f"Unknown PREDICTION_MODEL '{model}'. Must be 'ctchat' or 'medgemma'."
        )

    # Attach the full findings list to the result for the report builder
    result["findings"] = mask_metrics["findings"]

    return result
