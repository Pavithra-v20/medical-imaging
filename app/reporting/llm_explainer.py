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


def _build_prompt(prediction: dict, mask_metrics: dict) -> str:
    label = prediction.get("disease_label", "Unknown")
    confidence = float(prediction.get("confidence", 0.0)) * 100.0
    region = _build_region(mask_metrics)

    return f"""You are an expert radiologist. 2-3 line clinical note:

FINDING: {label.title()}
CONFIDENCE: {confidence:.0f}%
LOCATION: {region}

Professional summary. <120 words."""


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
def generate_llm_explanation(prediction: dict, mask_metrics: dict) -> Optional[str]:
    if not settings.explanation_enabled:
        return None
    if not settings.nim_api_key:
        logger.warning("nim_api_key_missing")
        return None

    prompt = _build_prompt(prediction, mask_metrics)

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
