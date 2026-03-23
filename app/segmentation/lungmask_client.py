import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


async def run_lungmask(nifti_path: str, session_id: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Run Lungmask (local) segmentation on a NIfTI file.
    Returns a tuple of (mask, spacing_mm).
    Lungmask outputs:
      0 = background
      1 = right lung
      2 = left lung
    """
    # Lazy imports to avoid heavy deps when not used
    import SimpleITK as sitk
    from lungmask import LMInferer

    image = sitk.ReadImage(str(nifti_path))
    if image.GetDimension() == 2:
        # Lungmask expects 3D images. Promote single-slice 2D to 3D.
        spacing = image.GetSpacing()
        image = sitk.JoinSeries(image)
        image.SetSpacing((spacing[0], spacing[1], 1.0))
        image.SetDirection((1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0))
    inferer = LMInferer(modelname="R231")
    mask = inferer.apply(image)
    spacing = image.GetSpacing()  # (x, y, z) in mm

    logger.info("lungmask_complete", session_id=session_id, mask_shape=getattr(mask, "shape", None))
    return mask, spacing
