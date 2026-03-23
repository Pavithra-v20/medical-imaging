import pydicom
from pathlib import Path
from app.utils.exceptions import IngestionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_dicom(dicom_path: Path) -> pydicom.Dataset:
    """
    Read and validate a DICOM file from disk.
    Checks that the file is a valid CT modality before returning.
    """
    try:
        ds = pydicom.dcmread(str(dicom_path))
    except Exception as e:
        raise IngestionError(f"Cannot read DICOM file: {e}")

    # Validate it is a CT scan
    modality = getattr(ds, "Modality", None)
    if modality != "CT":
        raise IngestionError(
            f"Expected CT modality, got '{modality}'. Only CT scans are supported."
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
