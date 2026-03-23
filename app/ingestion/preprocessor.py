import numpy as np
import nibabel as nib
from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_SPACING = (1.5, 1.5, 1.5)   # mm — VISTA-3D recommended input spacing
HU_MIN = -1000.0                    # Air
HU_MAX = 1000.0                     # Bone


def preprocess(nifti_path: Path) -> np.ndarray:
    """
    Load a NIfTI volume and normalize it for VISTA-3D:
      1. Clip Hounsfield Units to [HU_MIN, HU_MAX]
      2. Scale to [0, 1] range
      3. Add channel and batch dimensions → shape (1, 1, D, H, W)

    Returns a float32 numpy array ready to send to the segmentation model.
    """
    img = nib.load(str(nifti_path))
    volume = img.get_fdata(dtype=np.float32)

    # Clip HU range
    volume = np.clip(volume, HU_MIN, HU_MAX)

    # Normalize to [0, 1]
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)

    # Ensure 3D (D, H, W) before adding batch/channel
    if volume.ndim == 2:
        # Add a depth dimension: (H, W) → (1, H, W)
        volume = volume[np.newaxis, ...]
    
    # Add batch and channel dims: (D, H, W) → (1, 1, D, H, W)
    volume = volume[np.newaxis, np.newaxis, ...]

    logger.info(
        "volume_preprocessed",
        shape=volume.shape,
        min=float(volume.min()),
        max=float(volume.max()),
    )

    return volume
