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
    removed = []
    for tag in PHI_TAGS:
        if hasattr(ds, tag):
            setattr(ds, tag, "")
            removed.append(tag)

    logger.info("dicom_anonymized", stripped_tags=removed)
    return ds
