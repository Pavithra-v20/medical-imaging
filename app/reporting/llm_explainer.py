from __future__ import annotations

from typing import Optional

import tenacity
from openai import OpenAI

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _build_region(mask_metrics: dict) -> str:
    lesion_findings = mask_metrics.get("lesion_findings", [])
    if lesion_findings:
        top = lesion_findings[0]
        structure = top.get("structure_class", "unknown region")
        return structure.replace("_", " ")
    return "not specified"


def _build_prompt(prediction: dict, mask_metrics: dict, clinical_context: dict = None) -> str:
    label = prediction.get("disease_label", "Unknown")
    confidence = float(prediction.get("confidence", 0.0)) * 100.0
    region = _build_region(mask_metrics)
    lesion_count = mask_metrics.get("lesion_count", 0)
    max_size = mask_metrics.get("max_lesion_size_mm", 0.0)
    organ_count = mask_metrics.get("organ_count", 0)
    structures = [f.get("structure_class", "") for f in mask_metrics.get("findings", [])][:5]
    structures = [s.replace("_", " ") for s in structures if s]
    structures_str = ", ".join(structures) if structures else "none"

    clinical_info = ""
    if clinical_context:
        age = clinical_context.get("patient_age")
        sex = clinical_context.get("patient_sex")
        desc = clinical_context.get("study_description")
        if age or sex or desc:
            parts = []
            if age: parts.append(f"{age}-year-old")
            if sex: parts.append(sex)
            if desc: parts.append(f"presented for {desc}")
            clinical_info = "CLINICAL CONTEXT: " + " ".join(parts) + "\n"

    return f"""You are an expert radiologist. 2-3 line clinical note:
{clinical_info}
FINDING: {label.title()}
CONFIDENCE: {confidence:.0f}%
LOCATION: {region}
LESION_COUNT: {lesion_count}
MAX_LESION_SIZE_MM: {max_size:.1f}
ORGANS_DETECTED: {organ_count}
STRUCTURES: {structures_str}

Professional summary. Be concise and clinically precise. <120 words."""


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
def generate_llm_explanation(prediction: dict, mask_metrics: dict, clinical_context: dict = None) -> Optional[str]:
    if not settings.explanation_enabled:
        return None
    if not settings.nim_api_key:
        logger.warning("nim_api_key_missing")
        return None

    prompt = _build_prompt(prediction, mask_metrics, clinical_context)

    client = OpenAI(api_key=settings.nim_api_key, base_url=settings.nim_base_url)
    logger.info("nim_explanation_request", model=settings.nim_model)

    response = client.chat.completions.create(
        model=settings.nim_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=180,
        temperature=0.0,
    )

    explanation = response.choices[0].message.content.strip()
    return explanation or None
