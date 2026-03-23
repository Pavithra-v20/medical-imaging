from app.utils.logger import get_logger

logger = get_logger(__name__)


def summarize(prediction: dict, mask_metrics: dict, clinical_context: dict = None, rev_req: bool = False) -> str:
    """
    Generate a short plain-language summary of the session result.
    """
    disease_label = prediction.get("disease_label", "undetermined")
    confidence = prediction.get("confidence", 0.0)
    lesion_count = mask_metrics.get("lesion_count", 0)
    max_size = mask_metrics.get("max_lesion_size_mm", 0.0)
    organ_count = mask_metrics.get("organ_count", 0)

    # Clinical context intro
    intro = ""
    if clinical_context:
        age = clinical_context.get("patient_age")
        sex = clinical_context.get("patient_sex")
        if age and sex:
            intro = f"Scan for {age}Y {sex}. "
        elif age or sex:
            intro = f"Scan for {age or sex}. "

    # If no findings but we have a prediction, summarize based on prediction
    if organ_count == 0 and lesion_count == 0:
        sentence1 = "Image classification completed."
        sentence2 = (
            f"Primary finding: {disease_label} "
            f"(confidence: {confidence * 100:.0f}%)."
        )
        reasoning = prediction.get("reasoning", "")
        sentence3 = reasoning if reasoning else ""
        summary = " ".join(filter(None, [sentence1, sentence2, sentence3]))
        logger.info("summary_generated", length=len(summary))
        return summary

    # Sentence 1 — what was found
    if lesion_count > 0:
        lesion_names = list({
            f["structure_class"] for f in mask_metrics.get("lesion_findings", [])
        })
        finding_str = ", ".join(lesion_names[:3])
        sentence1 = (
            f"CT analysis identified {lesion_count} lesion(s) "
            f"({finding_str}), max size {max_size:.1f}mm."
        )
    else:
        sentence1 = (
            f"CT analysis identified {organ_count} organs "
            f"with no lesions detected."
        )

    # Sentence 2 — primary diagnosis
    sentence2 = (
        f"Primary finding: {disease_label} "
        f"(confidence: {confidence * 100:.0f}%)."
    )

    # Review flag
    warning = "⚠️ URGENT PHYSICIAN REVIEW RECOMMENDED." if rev_req else ""

    summary = " ".join(filter(None, [intro, sentence1, sentence2, warning]))

    logger.info("summary_generated", length=len(summary), rev_req=rev_req)
    return summary
