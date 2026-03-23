from app.utils.logger import get_logger

logger = get_logger(__name__)


def summarize(prediction: dict, mask_metrics: dict) -> str:
    """
    Generate a short plain-language summary (2-3 sentences) of the session result.
    This is stored in the 'summary' column and shown to the technician immediately
    after the scan is processed.

    Example output:
        "CT scan analysis identified 2 lung nodules, the largest measuring 14.2mm.
         Primary finding: lung nodule (confidence: 91%). Physician review is recommended."
    """
    disease_label = prediction.get("disease_label", "undetermined")
    confidence = prediction.get("confidence", 0.0)
    lesion_count = mask_metrics.get("lesion_count", 0)
    max_size = mask_metrics.get("max_lesion_size_mm", 0.0)
    organ_count = mask_metrics.get("organ_count", 0)

    # Sentence 1 — what was found
    if lesion_count > 0:
        lesion_names = list({
            f["structure_class"] for f in mask_metrics.get("lesion_findings", [])
        })
        finding_str = ", ".join(lesion_names[:3])
        sentence1 = (
            f"CT scan analysis identified {lesion_count} lesion(s) "
            f"({finding_str}), the largest measuring {max_size:.1f}mm."
        )
    else:
        sentence1 = (
            f"CT scan analysis identified {organ_count} anatomical structures "
            f"with no lesions detected."
        )

    # Sentence 2 — primary diagnosis
    sentence2 = (
        f"Primary finding: {disease_label} "
        f"(confidence: {confidence * 100:.0f}%)."
    )

    # Sentence 3 — review note
    reasoning = prediction.get("reasoning", "")
    sentence3 = reasoning if reasoning else ""

    summary = " ".join(filter(None, [sentence1, sentence2, sentence3]))

    logger.info("summary_generated", length=len(summary))
    return summary
