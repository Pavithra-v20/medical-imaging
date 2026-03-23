import numpy as np
from scipy import ndimage
from app.utils.logger import get_logger

logger = get_logger(__name__)

# VISTA-3D class label mapping (subset of 127 classes)
LABEL_MAP = {
    1:  "spleen",
    2:  "right_kidney",
    3:  "left_kidney",
    4:  "gallbladder",
    5:  "liver",
    6:  "stomach",
    7:  "aorta",
    8:  "pancreas",
    23: "lung_upper_lobe_left",
    24: "lung_lower_lobe_left",
    25: "lung_upper_lobe_right",
    26: "lung_middle_lobe_right",
    27: "lung_lower_lobe_right",
    28: "lung_nodule",
    29: "lung_tumor",
    30: "colon_tumor",
    31: "liver_tumor",
    32: "kidney_tumor",
    33: "pancreas_tumor",
    34: "bone_lesion",
}

# Labels we treat as lesions (not normal anatomy)
LESION_LABELS = {28, 29, 30, 31, 32, 33, 34}

# Size thresholds in mm — above these trigger rev_req
CRITICAL_SIZE_MM = 20.0
SUSPICIOUS_SIZE_MM = 6.0


def parse_masks(mask: np.ndarray, volume: np.ndarray) -> dict:
    """
    Parse the raw VISTA-3D integer mask into a structured dict of findings.

    For each detected label:
      - Compute 3D centroid (voxel coordinates)
      - Compute volume in cm³ (voxel count × voxel spacing)
      - For lesions, compute bounding-box longest axis in mm
      - Run connected-components to count discrete lesion instances

    Returns a dict ready to pass to the predictor and report builder.
    """
    voxel_spacing_mm = np.array([1.5, 1.5, 1.5])  # matches preprocessor target spacing
    voxel_volume_cm3 = float(np.prod(voxel_spacing_mm) / 1000.0)

    findings = []
    present_labels = np.unique(mask)
    present_labels = present_labels[present_labels != 0]  # exclude background

    for label_id in present_labels:
        label_id = int(label_id)
        class_name = LABEL_MAP.get(label_id, f"unknown_class_{label_id}")
        binary_mask = (mask == label_id).astype(np.uint8)

        # Voxel count and volume
        voxel_count = int(binary_mask.sum())
        volume_cm3 = round(voxel_count * voxel_volume_cm3, 2)

        # 3D centroid
        centroid = ndimage.center_of_mass(binary_mask)
        centroid_voxel = [int(c) for c in centroid]

        # Bounding box and size in mm
        coords = np.argwhere(binary_mask)
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        bbox_size_mm = (bbox_max - bbox_min) * voxel_spacing_mm
        size_mm = round(float(bbox_size_mm.max()), 2)

        # Lesion instance count via connected components
        if label_id in LESION_LABELS:
            labeled_array, num_instances = ndimage.label(binary_mask)
        else:
            num_instances = 1

        finding = {
            "label_id": label_id,
            "structure_class": class_name,
            "is_lesion": label_id in LESION_LABELS,
            "voxel_count": voxel_count,
            "volume_cm3": volume_cm3,
            "size_mm": size_mm,
            "location_voxel": centroid_voxel,
            "num_instances": num_instances,
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
        }
        findings.append(finding)

    lesion_findings = [f for f in findings if f["is_lesion"]]
    max_lesion_size = max((f["size_mm"] for f in lesion_findings), default=0.0)

    result = {
        "findings": findings,
        "lesion_findings": lesion_findings,
        "organ_count": len(findings) - len(lesion_findings),
        "lesion_count": len(lesion_findings),
        "max_lesion_size_mm": max_lesion_size,
        "has_critical_lesion": max_lesion_size >= CRITICAL_SIZE_MM,
        "has_suspicious_lesion": max_lesion_size >= SUSPICIOUS_SIZE_MM,
    }

    logger.info(
        "masks_parsed",
        organ_count=result["organ_count"],
        lesion_count=result["lesion_count"],
        max_lesion_size_mm=max_lesion_size,
    )

    return result


def parse_lungmask_masks(mask: np.ndarray, spacing_xyz_mm: tuple[float, float, float]) -> dict:
    """
    Parse Lungmask output (labels: 1=right lung, 2=left lung) into mask metrics.
    spacing_xyz_mm is in (x, y, z) order from the NIfTI header.
    """
    # Convert spacing to match mask array axes (z, y, x)
    voxel_spacing_mm = np.array([spacing_xyz_mm[2], spacing_xyz_mm[1], spacing_xyz_mm[0]])
    voxel_volume_cm3 = float(np.prod(voxel_spacing_mm) / 1000.0)

    label_map = {
        1: "right_lung",
        2: "left_lung",
    }

    findings = []
    present_labels = np.unique(mask)
    present_labels = present_labels[present_labels != 0]

    for label_id in present_labels:
        label_id = int(label_id)
        class_name = label_map.get(label_id, f"unknown_class_{label_id}")
        binary_mask = (mask == label_id).astype(np.uint8)

        voxel_count = int(binary_mask.sum())
        volume_cm3 = round(voxel_count * voxel_volume_cm3, 2)

        centroid = ndimage.center_of_mass(binary_mask)
        centroid_voxel = [int(c) for c in centroid]

        coords = np.argwhere(binary_mask)
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        bbox_size_mm = (bbox_max - bbox_min) * voxel_spacing_mm
        size_mm = round(float(bbox_size_mm.max()), 2)

        finding = {
            "label_id": label_id,
            "structure_class": class_name,
            "is_lesion": False,
            "voxel_count": voxel_count,
            "volume_cm3": volume_cm3,
            "size_mm": size_mm,
            "location_voxel": centroid_voxel,
            "num_instances": 1,
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
        }
        findings.append(finding)

    result = {
        "findings": findings,
        "lesion_findings": [],
        "organ_count": len(findings),
        "lesion_count": 0,
        "max_lesion_size_mm": 0.0,
        "has_critical_lesion": False,
        "has_suspicious_lesion": False,
    }

    logger.info(
        "lungmask_parsed",
        organ_count=result["organ_count"],
        lesion_count=result["lesion_count"],
    )

    return result


def parse_image_mask(mask: np.ndarray, metrics: dict) -> dict:
    """
    Parse a 2D image mask into the common mask_metrics structure.
    We treat a sufficiently large foreground region as a single "lesion".
    """
    area_ratio = metrics.get("area_ratio", 0.0)
    has_lesion = area_ratio >= 0.02

    findings = []
    lesion_findings = []
    if has_lesion:
        lesion = {
            "label_id": 1,
            "structure_class": "image_region",
            "is_lesion": True,
            "voxel_count": int(mask.sum()),
            "volume_cm3": 0.0,
            "size_mm": float(metrics.get("size_px", 0.0)),
            "location_voxel": metrics.get("centroid", [0, 0]),
            "num_instances": 1,
            "bbox_min": metrics.get("bbox_min", [0, 0]),
            "bbox_max": metrics.get("bbox_max", [0, 0]),
        }
        lesion_findings.append(lesion)
        findings.append(lesion)

    result = {
        "findings": findings,
        "lesion_findings": lesion_findings,
        "organ_count": 0,
        "lesion_count": 1 if has_lesion else 0,
        "max_lesion_size_mm": float(metrics.get("size_px", 0.0)) if has_lesion else 0.0,
        "has_critical_lesion": False,
        "has_suspicious_lesion": has_lesion,
        "segmentation_method": "Otsu threshold (2D image)",
    }

    logger.info("image_mask_parsed", lesion_count=result["lesion_count"], area_ratio=area_ratio)
    return result
