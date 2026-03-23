import numpy as np
import tritonclient.http as httpclient
from app.config import get_settings
from app.utils.exceptions import SegmentationError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def run_triton_inference(
    volume: np.ndarray,
    model_name: str = "vista3d",
    input_name: str = "INPUT__0",
    output_name: str = "OUTPUT__0",
) -> np.ndarray:
    """
    Send a preprocessed volume to Triton Inference Server and retrieve
    the segmentation mask. Used as an alternative to the NIM cloud API
    when running VISTA-3D on a local GPU via Triton.

    Args:
        volume:      float32 array of shape (1, 1, D, H, W)
        model_name:  Triton model repository name
        input_name:  Triton model input tensor name
        output_name: Triton model output tensor name

    Returns:
        Integer mask array of shape (D, H, W)
    """
    try:
        client = httpclient.InferenceServerClient(url=settings.triton_url)
    except Exception as e:
        raise SegmentationError(f"Cannot connect to Triton at {settings.triton_url}: {e}")

    if not client.is_model_ready(model_name):
        raise SegmentationError(f"Model '{model_name}' is not ready on Triton server.")

    # Prepare input tensor
    inputs = [
        httpclient.InferInput(input_name, volume.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(volume.astype(np.float32))

    # Prepare output tensor
    outputs = [httpclient.InferRequestedOutput(output_name)]

    try:
        response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    except Exception as e:
        raise SegmentationError(f"Triton inference failed: {e}")

    mask = response.as_numpy(output_name)

    logger.info(
        "triton_inference_complete",
        model=model_name,
        input_shape=volume.shape,
        output_shape=mask.shape,
    )

    # Remove batch dim if present: (1, D, H, W) → (D, H, W)
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    return mask
