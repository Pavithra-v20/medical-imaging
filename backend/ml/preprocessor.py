"""
Image preprocessing utilities for the Medical AI Diagnostic System.

Matches the exact preprocessing pipeline used during model training
(see notebook cell: `preprocess_image`):

  1. Decode image bytes → RGB tensor
  2. Resize to 224×224 (Config.IMG_SIZE)
  3. Cast to float32 and scale to [0, 1]
  4. Normalise to [-1, 1]  →  (pixel / 255 - 0.5) × 2

Returns a NumPy array of shape (1, 224, 224, 3) ready for `model.predict()`.
"""
import logging
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMG_SIZE: int = 224


def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a normalised NumPy batch tensor.

    Args:
        image_bytes: Raw bytes of the uploaded image (JPEG, PNG, etc.).

    Returns:
        np.ndarray: Shape (1, 224, 224, 3), dtype float32, values in [-1, 1].

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)                # (224, 224, 3)
    arr = arr / 255.0                                     # scale to [0, 1]
    arr = (arr - 0.5) * 2.0                               # normalise to [-1, 1]
    arr = np.expand_dims(arr, axis=0)                     # (1, 224, 224, 3)

    logger.debug("Preprocessed image to shape %s, range [%.2f, %.2f]",
                 arr.shape, arr.min(), arr.max())
    return arr


def preprocess_file(image_path: str) -> np.ndarray:
    """
    Load an image from disk and return a normalised batch tensor.

    Args:
        image_path: Absolute or relative path to an image file.

    Returns:
        np.ndarray: Shape (1, 224, 224, 3), dtype float32, values in [-1, 1].
    """
    with open(image_path, "rb") as f:
        return preprocess_bytes(f.read())


def tensor_to_pil(tensor: np.ndarray) -> Image.Image:
    """
    Convert a preprocessed tensor back to a PIL image for visualisation.

    Reverses the normalisation so the image is displayable.

    Args:
        tensor: NumPy array of shape (H, W, 3) or (1, H, W, 3) in [-1, 1].

    Returns:
        PIL.Image.Image: RGB image.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]                          # drop batch dim
    rgb = ((tensor / 2.0 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb)
