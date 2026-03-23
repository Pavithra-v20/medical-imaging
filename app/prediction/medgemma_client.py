import httpx
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.exceptions import PredictionError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a radiology AI assistant specializing in CT scan analysis.
Analyze the provided CT segmentation findings and return ONLY a JSON object with:
- disease_label: the primary diagnosis (string)
- confidence: your confidence between 0.0 and 1.0
- reasoning: brief clinical reasoning (1-2 sentences)

Respond with valid JSON only. No markdown, no preamble."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_medgemma(mask_metrics: dict) -> dict:
    """
    Send segmentation findings to MedGemma 1.5 for disease prediction.

    MedGemma 1.5 is Google's medical multimodal model supporting CT, MRI,
    dermatology, and more. Used as an alternative to CT-CHAT for broader
    multi-disease coverage.

    Returns a dict with disease_label, confidence, reasoning.
    """
    user_content = f"""CT segmentation findings:
{json.dumps(mask_metrics, indent=2)}

Based on these findings, provide your primary diagnosis."""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\n" + user_content}
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512,
        },
    }

    url = f"{settings.medgemma_url}:generateContent?key={settings.google_api_key}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise PredictionError(f"MedGemma returned {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        raise PredictionError(f"MedGemma request failed: {e}")

    raw_text = (
        response.json()["candidates"][0]["content"]["parts"][0]["text"]
    )

    # Strip any accidental markdown fences
    clean = raw_text.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        raise PredictionError(f"MedGemma returned non-JSON response: {raw_text}")

    result["raw_response"] = raw_text
    result["model_used"] = "medgemma"

    logger.info(
        "medgemma_prediction",
        disease_label=result.get("disease_label"),
        confidence=result.get("confidence"),
    )

    return result
