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


import base64
import io
import numpy as np
from PIL import Image

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_medgemma(mask_metrics: dict, volume: np.ndarray | None = None) -> dict:
    """
    Send segmentation findings and a representative CT slice to MedGemma 1.5.
    """
    user_content = f"""CT segmentation findings:
{json.dumps(mask_metrics, indent=2)}

Based on these findings and the provided image, provide your primary diagnosis."""

    parts = [{"text": SYSTEM_PROMPT + "\n\n" + user_content}]

    if volume is not None and isinstance(volume, np.ndarray) and volume.ndim == 3:
        # Extract middle slice as representative
        mid_idx = volume.shape[0] // 2
        slice_arr = volume[mid_idx]
        
        # Normalize to 0-255 for PNG
        vmin, vmax = slice_arr.min(), slice_arr.max()
        if vmax > vmin:
            slice_arr = ((slice_arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            slice_arr = slice_arr.astype(np.uint8)
            
        img = Image.fromarray(slice_arr)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": img_b64
            }
        })

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
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
