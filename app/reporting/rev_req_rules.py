from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Disease labels that always require physician review regardless of confidence
ALWAYS_REVIEW_LABELS = {
    "lung_tumor",
    "liver_tumor",
    "pancreas_tumor",
    "kidney_tumor",
    "colon_tumor",
    "bone_lesion",
    "metastasis",
    "malignancy",
    "cancer",
    "carcinoma",
}


def evaluate_rev_req(prediction: dict, mask_metrics: dict) -> bool:
    """
    Determine whether the session requires physician review (rev_req = True).

    Rules (any one triggers rev_req):
      1. Model confidence is below the configured threshold
      2. A critical lesion is present (size >= 20mm)
      3. Multiple lesion instances detected (possible metastatic pattern)
      4. Predicted disease label is in the always-review set
      5. No lesions found but organs show abnormal volumes (future rule placeholder)
    """
    confidence = prediction.get("confidence", 0.0)
    disease_label = prediction.get("disease_label", "").lower()

    # Rule 1: Low confidence
    if confidence < settings.confidence_threshold:
        logger.info("rev_req_low_confidence", confidence=confidence)
        return True

    # Rule 2: Critical lesion size
    if mask_metrics.get("has_critical_lesion", False):
        logger.info("rev_req_critical_lesion", size_mm=mask_metrics["max_lesion_size_mm"])
        return True

    # Rule 3: Multiple lesion instances in any single class
    for finding in mask_metrics.get("lesion_findings", []):
        if finding.get("num_instances", 1) > 1:
            logger.info("rev_req_multiple_lesions", structure=finding["structure_class"])
            return True

    # Rule 4: Always-review disease label
    for keyword in ALWAYS_REVIEW_LABELS:
        if keyword in disease_label:
            logger.info("rev_req_critical_label", label=disease_label)
            return True

    # Rule 5: Suspicious keywords in reasoning
    reasoning = prediction.get("reasoning", "").lower()
    if any(k in reasoning for k in ["suspicious", "malignant", "suggestive of", "concerning"]):
        logger.info("rev_req_suspicious_reasoning")
        return True

    return False
