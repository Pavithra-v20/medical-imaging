import pydicom
from app.utils.logger import get_logger

logger = get_logger(__name__)

# DICOM tags that contain patient-identifying information (PHI)
PHI_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "InstitutionName",
    "InstitutionAddress",
    "AccessionNumber",
    "StudyID",
]


def anonymize(ds: pydicom.Dataset) -> pydicom.Dataset:
    """
    Strip all PHI tags from a DICOM dataset in-place.
    Replaces values with empty strings rather than deleting the tags
    so downstream tools do not fail on missing tags.
    """
    phi_tags = [
        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
        "PatientAge", "PatientAddress", "PatientTelephoneNumbers",
        "ReferringPhysicianName", "InstitutionName", "InstitutionAddress",
        "AccessionNumber", "StudyID"
    ]
    
    removed = []
    for tag in phi_tags:
        if hasattr(ds, tag):
            setattr(ds, tag, "")
            removed.append(tag)

    return ds


def anonymize_series(series: list[pydicom.Dataset]) -> list[pydicom.Dataset]:
    """
    Anonymize all slices in a DICOM series.
    """
    for ds in series:
        anonymize(ds)
    
    logger.info("dicom_series_anonymized", count=len(series))
    return series
