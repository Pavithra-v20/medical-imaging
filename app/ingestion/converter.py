import pydicom
import nibabel as nib
import numpy as np
from pathlib import Path
from app.utils.exceptions import IngestionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def convert_to_nifti(ds: pydicom.Dataset, output_dir: str, session_id: str) -> Path:
    """
    Convert a DICOM dataset to NIfTI format (.nii.gz).
    Extracts the pixel array, applies rescale slope/intercept to get
    Hounsfield Units, builds an affine matrix from DICOM spatial metadata,
    and writes out a NIfTI file ready for VISTA-3D.
    """
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise IngestionError(f"Cannot extract pixel data: {e}")

    # Apply Hounsfield Unit rescaling
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    # Build affine from pixel spacing and image position
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
    voxel_sizes = [float(pixel_spacing[0]), float(pixel_spacing[1]), slice_thickness]

    affine = np.diag([*voxel_sizes, 1.0])

    nifti_img = nib.Nifti1Image(pixel_array, affine)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nifti_path = output_path / f"{session_id}.nii.gz"

    nib.save(nifti_img, str(nifti_path))
    logger.info("nifti_saved", path=str(nifti_path), shape=pixel_array.shape)

    return nifti_path
