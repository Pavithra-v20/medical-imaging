import pydicom
import nibabel as nib
import numpy as np
from pathlib import Path
from app.utils.exceptions import IngestionError
from app.utils.logger import get_logger

import SimpleITK as sitk

logger = get_logger(__name__)


def convert_to_nifti(ds: pydicom.Dataset, output_dir: str, session_id: str) -> Path:
    """
    Convert a single DICOM dataset to NIfTI format (.nii.gz).
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
    pixel_spacing = getattr(ds, "PixelSpacing", [1.5, 1.5])
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


def convert_series_to_nifti(
    slices: list[pydicom.Dataset], 
    output_dir: str, 
    session_id: str
) -> Path:
    """
    Stack a sorted list of DICOM slices into a 3D NIfTI volume.
    Applies rescale slope/intercept from the first slice.
    (Legacy path; prefer convert_dicom_dir_to_nifti for better spacing/orientation.)
    """
    if not slices:
        raise IngestionError("No slices provided for NIfTI conversion")

    # Stack along axis=0 -> (D, H, W)
    try:
        volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
    except Exception as e:
        raise IngestionError(f"Failed to stack DICOM slices: {e}")

    # Apply HU rescaling from the first slice
    first_slice = slices[0]
    slope = float(getattr(first_slice, "RescaleSlope", 1))
    intercept = float(getattr(first_slice, "RescaleIntercept", 0))
    volume = volume * slope + intercept

    # Build affine from PixelSpacing and SliceThickness
    pixel_spacing = getattr(first_slice, "PixelSpacing", [1.5, 1.5])
    slice_thickness = float(getattr(first_slice, "SliceThickness", 1.0))
    voxel_sizes = [slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1])]

    affine = np.diag([*voxel_sizes, 1.0])
    nifti_img = nib.Nifti1Image(volume, affine)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nifti_path = output_path / f"{session_id}.nii.gz"

    nib.save(nifti_img, str(nifti_path))
    logger.info("series_nifti_saved", path=str(nifti_path), shape=volume.shape)

    return nifti_path


def convert_dicom_dir_to_nifti(
    dicom_dir: Path,
    output_dir: str,
    session_id: str,
    hu_window: tuple[int, int] = (-1000, 400),
) -> Path:
    """
    Convert a DICOM series directory to NIfTI using SimpleITK to preserve
    spacing and orientation. Applies optional HU windowing.
    """
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if not series_ids:
            raise IngestionError(f"No DICOM series found in {dicom_dir}")

        series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        reader.SetFileNames(series_files)
        image = reader.Execute()
    except Exception as e:
        raise IngestionError(f"Failed to read DICOM series with SimpleITK: {e}")

    if hu_window:
        low, high = hu_window
        image = sitk.Clamp(image, lowerBound=low, upperBound=high)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nifti_path = output_path / f"{session_id}.nii.gz"

    sitk.WriteImage(image, str(nifti_path))
    logger.info("series_nifti_saved_sitk", path=str(nifti_path), window=hu_window)
    return nifti_path
