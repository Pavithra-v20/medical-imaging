class CTAgentError(Exception):
    """Base exception for all agent errors."""
    pass


class IngestionError(CTAgentError):
    """Raised when DICOM loading, anonymization, or conversion fails."""
    pass


class SegmentationError(CTAgentError):
    """Raised when VISTA-3D segmentation fails."""
    pass


class PredictionError(CTAgentError):
    """Raised when CT-CHAT or MedGemma prediction fails."""
    pass


class ReportingError(CTAgentError):
    """Raised when report generation fails."""
    pass


class DatabaseError(CTAgentError):
    """Raised when PostgreSQL write fails."""
    pass
