import pydicom
from pathlib import Path
from app.utils.exceptions import IngestionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_dicom(dicom_path: Path) -> pydicom.Dataset:
    """
    Read and validate a DICOM file from disk.
    Checks that the file is a valid CT or MR modality before returning.
    """
    try:
        ds = pydicom.dcmread(str(dicom_path))
    except Exception as e:
        raise IngestionError(f"Cannot read DICOM file: {e}")

    # Validate it is a CT or MR scan
    modality = getattr(ds, "Modality", None)
    if modality not in ["CT", "MR"]:
        raise IngestionError(
            f"Expected CT or MR modality, got '{modality}'. Only CT and MRI scans are supported."
        )

    logger.info(
        "dicom_loaded",
        path=str(dicom_path),
        modality=modality,
        rows=getattr(ds, "Rows", None),
        columns=getattr(ds, "Columns", None),
        slices=getattr(ds, "NumberOfFrames", "N/A"),
    )

    return ds


def load_dicom_series(dicom_dir: Path) -> list[pydicom.Dataset]:
    """
    Read all DICOM files in a directory and return a sorted list of datasets.
    Validates that each file is a valid CT modality.
    """
    datasets = []
    for file_path in dicom_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(file_path))
            # Validate CT modality
            modality = getattr(ds, "Modality", None)
            if modality != "CT":
                logger.warning("invalid_modality_in_series", path=str(file_path), modality=modality)
                continue
            datasets.append(ds)
        except Exception as e:
            logger.warning("skipping_invalid_dicom", path=str(file_path), error=str(e))

    if not datasets:
        raise IngestionError(f"No valid CT DICOM files found in {dicom_dir}")

    # Sort by SliceLocation, fallback to InstanceNumber
    datasets.sort(key=lambda x: (float(getattr(x, "SliceLocation", 0)), int(getattr(x, "InstanceNumber", 0))))

    logger.info("dicom_series_loaded", directory=str(dicom_dir), count=len(datasets))
    return datasets
