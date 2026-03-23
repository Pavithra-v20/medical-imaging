import httpx
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.exceptions import PredictionError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a radiology AI assistant. You will receive structured
findings from a CT scan segmentation model. Analyze the findings and return a
JSON object with:
- disease_label: the primary diagnosis (string)
- confidence: your confidence score between 0.0 and 1.0
- reasoning: brief clinical reasoning (1-2 sentences)

Respond ONLY with a valid JSON object. No markdown, no preamble."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_ct_chat(mask_metrics: dict) -> dict:
    """
    Send segmentation findings to CT-CHAT for disease prediction.

    CT-CHAT is a medical LLM trained on CT-RATE (2.7M QA pairs from CT volumes).
    It receives the structured mask metrics JSON and returns a disease label
    with confidence and reasoning.

    Returns a dict with disease_label, confidence, reasoning.
    """
    user_content = f"""CT segmentation findings:
{json.dumps(mask_metrics, indent=2)}

Based on these findings, what is the primary diagnosis?"""

    payload = {
        "model": "ct-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                settings.ctchat_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise PredictionError(f"CT-CHAT returned {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        raise PredictionError(f"CT-CHAT request failed: {e}")

    raw_text = response.json()["choices"][0]["message"]["content"]

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        raise PredictionError(f"CT-CHAT returned non-JSON response: {raw_text}")

    result["raw_response"] = raw_text
    result["model_used"] = "ctchat"

    logger.info(
        "ctchat_prediction",
        disease_label=result.get("disease_label"),
        confidence=result.get("confidence"),
    )

    return result
