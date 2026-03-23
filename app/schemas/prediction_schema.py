from pydantic import BaseModel
from typing import Optional


class Finding(BaseModel):
    structure_class: str
    location_voxel: list[int]        # [x, y, z] centroid
    size_mm: Optional[float] = None  # bounding box longest axis
    volume_cm3: Optional[float] = None
    confidence: float                # softmax score 0–1


class PredictionResult(BaseModel):
    disease_label: str
    confidence: float
    findings: list[Finding]
    model_used: str                  # "ctchat" or "medgemma"
    raw_response: Optional[str] = None
