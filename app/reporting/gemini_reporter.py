from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

TEMPLATE_PATH = Path(__file__).with_name("report_template.md")


def _load_template() -> str:
    try:
        return TEMPLATE_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("report_template_load_failed", error=str(e))
        return ""


def _build_prompt(
    summary: str,
    prediction: dict,
    mask_metrics: dict,
    session_id: str,
    technician_id: str,
    patient_id: str,
    physician_id: str,
    clinical_context: dict | None = None,
) -> str:
    template = _load_template()
    disease_label = prediction.get("disease_label", "Unknown")
    confidence = prediction.get("confidence", 0.0)
    model_used = prediction.get("model_used", "unknown")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    findings = mask_metrics.get("findings", [])
    key_findings = [
        f"{f.get('structure_class', 'unknown')} (size {f.get('size_mm', 0):.1f}mm)"
        for f in findings[:5]
    ]
    key_findings_text = ", ".join(key_findings) if key_findings else "None detected"

    prompt = f"""You are a radiology assistant. Fill the report template below.
Use the provided summary and structured data. Output MUST be valid Markdown
and must follow the template headings exactly.

TEMPLATE:
{template}

DATA:
- session_id: {session_id}
- technician_id: {technician_id}
- patient_id: {patient_id}
- physician_id: {physician_id}
- generated_at: {generated_at}
- summary: {summary}
- disease_label: {disease_label}
- confidence: {confidence}
- model_used: {model_used}
- key_findings: {key_findings_text}
- patient_age: {clinical_context.get('patient_age') if clinical_context else 'N/A'}
- patient_sex: {clinical_context.get('patient_sex') if clinical_context else 'N/A'}
- study_description: {clinical_context.get('study_description') if clinical_context else 'N/A'}

LUNG-RADS GUIDANCE:
If findings involve lung nodules, categorize them according to Lung-RADS 1.1 criteria.
Place the category (e.g., '1', '2', '3', '4A', '4B', '4X') in the {{lung_rads}} placeholder.
If not applicable, use 'N/A'.

Return the filled template only. Do not add extra sections.
"""
    return prompt


def _image_part(image_path: str) -> Optional[dict]:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    mime = "image/png"
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif path.suffix.lower() == ".webp":
        mime = "image/webp"

    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": mime, "data": data}}


async def generate_gemini_report(
    summary: str,
    prediction: dict,
    mask_metrics: dict,
    session_id: str,
    technician_id: str,
    patient_id: str,
    physician_id: str,
    image_path: Optional[str] = None,
    clinical_context: dict | None = None,
) -> Optional[str]:
    if not settings.report_enabled:
        return None
    if not settings.google_api_key:
        logger.warning("gemini_api_key_missing")
        return None

    prompt = _build_prompt(
        summary=summary,
        prediction=prediction,
        mask_metrics=mask_metrics,
        session_id=session_id,
        technician_id=technician_id,
        patient_id=patient_id,
        physician_id=physician_id,
        clinical_context=clinical_context,
    )

    parts = [{"text": prompt}]
    img_part = _image_part(image_path) if image_path else None
    if img_part:
        parts.append(img_part)

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    url = f"{settings.gemini_base_url}/{settings.gemini_model}:generateContent?key={settings.google_api_key}"
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("gemini_report_http_error", status=e.response.status_code, detail=e.response.text)
        return None
    except httpx.RequestError as e:
        logger.error("gemini_report_request_failed", detail=str(e))
        return None

    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return text.strip() if text else None
