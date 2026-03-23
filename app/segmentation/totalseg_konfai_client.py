from __future__ import annotations

from pathlib import Path
import subprocess
import numpy as np
import nibabel as nib

from app.config import get_settings
from app.utils.exceptions import SegmentationError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _find_labelmap(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*.nii*"))
    if not candidates:
        raise SegmentationError(f"No NIfTI outputs found in {output_dir}")
    # Prefer files that look like a single labelmap
    for name in ["segmentations.nii.gz", "segmentations.nii", "segmentation.nii.gz", "segmentation.nii"]:
        for c in candidates:
            if c.name.lower() == name:
                return c
    return candidates[0]


def run_totalseg_konfai(nifti_path: str, session_id: str) -> np.ndarray:
    """
    Run TotalSegmentator (konfai) locally on CPU.
    Returns a labelmap as a numpy array.
    """
    output_dir = Path(settings.storage_masks) / f"totalseg_{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "totalsegmentator-konfai",
        settings.totalseg_task,
        "-i",
        str(nifti_path),
        "-o",
        str(output_dir),
        "--cpu",
        str(settings.totalseg_cpu),
    ]

    logger.info("totalseg_start", task=settings.totalseg_task, cpu=settings.totalseg_cpu)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise SegmentationError("totalsegmentator-konfai not installed or not on PATH") from e
    except subprocess.CalledProcessError as e:
        raise SegmentationError(f"TotalSegmentator failed: {e.stderr or e.stdout}") from e

    labelmap_path = _find_labelmap(output_dir)
    img = nib.load(str(labelmap_path))
    mask = img.get_fdata().astype(np.int16)

    logger.info("totalseg_complete", labelmap=str(labelmap_path), shape=mask.shape)
    return mask
