import httpx
import numpy as np
import base64
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.exceptions import SegmentationError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def run_segmentation(volume: np.ndarray, session_id: str) -> np.ndarray:
    """
    Send a preprocessed CT volume to the NVIDIA VISTA-3D NIM endpoint.
    Returns a 3D integer mask array where each voxel holds a class label (0–127).

    Retries up to 3 times with exponential backoff on transient failures.
    """
    # Encode volume as base64 for JSON transport
    volume_bytes = volume.astype(np.float32).tobytes()
    volume_b64 = base64.b64encode(volume_bytes).decode("utf-8")

    payload = {
        "input": {
            "data": volume_b64,
            "shape": list(volume.shape),
            "dtype": "float32",
        },
        "inference_mode": "everything",  # segment all 127 classes
    }

    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                settings.vista3d_nim_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise SegmentationError(f"VISTA-3D returned {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        raise SegmentationError(f"VISTA-3D request failed: {e}")

    result = response.json()

    # Decode the returned mask
    mask_b64 = result["output"]["mask"]
    mask_bytes = base64.b64decode(mask_b64)
    mask_shape = result["output"]["shape"]
    mask = np.frombuffer(mask_bytes, dtype=np.int16).reshape(mask_shape)

    logger.info(
        "segmentation_complete",
        session_id=session_id,
        mask_shape=mask_shape,
        unique_labels=int(np.unique(mask).size),
    )

    return mask
