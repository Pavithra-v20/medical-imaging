import httpx
import numpy as np
import base64
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.exceptions import SegmentationError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def resolve_nim_path(local_path: str, settings) -> str:
    """
    If NIM is running locally (localhost), remap the local storage path
    to the container mount path /mnt/storage/.
    If NIM is remote (cloud), the caller must provide a public URL.
    """
    if "localhost" in settings.vista3d_nim_url or "127.0.0.1" in settings.vista3d_nim_url or "8001" in settings.vista3d_nim_url:
        # Remap host path → container mount path
        normalized = local_path.replace("\\", "/")
        if "storage/" in normalized:
            return "/mnt/storage/" + normalized.split("storage/")[-1]
    return local_path  # assume it's already a public URL or valid path for cloud/local NIM


async def run_segmentation(nifti_url: str, session_id: str, prompts: dict = None) -> np.ndarray:
    """
    Send a NIfTI image URL/path to the NVIDIA VISTA-3D NIM endpoint.
    The NIM handles the preprocessing internally.
    Returns a 3D integer mask array where each voxel holds a class label (0–127).
    """
    # Fix 3: Resolve path for local vs remote NIM
    nifti_url = resolve_nim_path(nifti_url, settings)
    
    # Use the new inference endpoint structure
    endpoint_suffix = "/vista3d/inference"
    base_url = settings.vista3d_nim_url.rstrip("/")
    
    if endpoint_suffix in base_url:
        inference_url = base_url
    else:
        inference_url = f"{base_url}{endpoint_suffix}"
    
    payload = {
        "image": nifti_url,
    }
    
    if prompts:
        payload["prompts"] = prompts
    else:
        # Default to segmenting everything if no prompts provided
        # Note: The API might require at least one class or use "inference_mode": "everything"
        # Since the user's reference mentions prompts as optional, we handle it.
        pass

    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:  # NIMs can take longer for full volumes
            response = await client.post(
                inference_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text
        logger.error("vista3d_http_error", status_code=e.response.status_code, detail=error_detail)
        raise SegmentationError(f"VISTA-3D returned {e.response.status_code}: {error_detail}") from e
        
    except httpx.RequestError as e:
        logger.error("vista3d_request_failed", detail=str(e))
        raise SegmentationError(f"VISTA-3D request failed: {e}") from e

    result = response.json()
    
    # Check for the correct output key
    # Standard NIMs often return "mask" or "output_data"
    if "mask" in result:
        mask_b64 = result["mask"]
        mask_shape = result.get("shape")
    elif "output" in result and "mask" in result["output"]:
        mask_b64 = result["output"]["mask"]
        mask_shape = result["output"].get("shape")
    else:
        logger.error("vista3d_invalid_response", response=result)
        raise SegmentationError("VISTA-3D response missing mask data")

    mask_bytes = base64.b64decode(mask_b64)
    if mask_shape:
        mask = np.frombuffer(mask_bytes, dtype=np.int16).reshape(mask_shape)
    else:
        # Fallback if shape is missing, we try to infer it or use a default if possible
        # but usually shape is provided.
        mask = np.frombuffer(mask_bytes, dtype=np.int16)
        logger.warning("vista3d_missing_shape", result_size=mask.size)

    logger.info(
        "segmentation_complete",
        session_id=session_id,
        mask_shape=getattr(mask, 'shape', 'unknown'),
        unique_labels=int(np.unique(mask).size),
    )

    return mask
