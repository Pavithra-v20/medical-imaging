import numpy as np
from PIL import Image
from scipy import ndimage

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _otsu_threshold(image: np.ndarray) -> int:
    """Compute Otsu threshold for a uint8 grayscale image."""
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total = image.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 0

    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = i

    return int(threshold)


def segment_image(path: str) -> tuple[np.ndarray, dict]:
    """
    Segment a 2D image using Otsu thresholding + morphology.
    Returns (mask, metrics) where mask is uint8 {0,1}.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    thresh = _otsu_threshold(arr)
    mask = (arr > thresh).astype(np.uint8)

    # If mask is almost full, invert
    if mask.mean() > 0.9:
        mask = (1 - mask).astype(np.uint8)

    # Clean small noise
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3))).astype(np.uint8)
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3))).astype(np.uint8)

    area_ratio = float(mask.mean())
    coords = np.argwhere(mask == 1)
    if coords.size == 0:
        bbox_min = [0, 0]
        bbox_max = [0, 0]
        centroid = [0, 0]
        size_px = 0.0
    else:
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        centroid = ndimage.center_of_mass(mask)
        size_px = float(max(bbox_max - bbox_min))

    metrics = {
        "area_ratio": area_ratio,
        "bbox_min": bbox_min.tolist() if hasattr(bbox_min, "tolist") else bbox_min,
        "bbox_max": bbox_max.tolist() if hasattr(bbox_max, "tolist") else bbox_max,
        "centroid": [int(c) for c in centroid],
        "size_px": size_px,
    }

    logger.info("image_segmented", area_ratio=area_ratio, size_px=size_px)
    return mask, metrics
