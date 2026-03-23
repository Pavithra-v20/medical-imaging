from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from backend.agents.state import DiagnosticState
from backend.config import settings

logger = logging.getLogger(__name__)


def _parse_classes(value: str) -> List[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def _load_state_dict(path: str) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError("Unsupported model checkpoint format")

    # Strip DataParallel prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    return cleaned


@lru_cache(maxsize=1)
def _get_model():
    model_path = getattr(settings, "mr_model_path", settings.MODEL_PT_PATH)
    if not model_path:
        raise ValueError("MODEL_PT_PATH not set")

    mr_classes = getattr(settings, "mr_classes", "glioma, meningioma, no_tumor, pituitary")
    classes = _parse_classes(mr_classes)
    num_classes = len(classes)

    # Default to resnet18 unless the checkpoint requires a different arch
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state_dict = _load_state_dict(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, classes


def _preprocess(image: Image.Image) -> torch.Tensor:
    # Convert to RGB and resize to 224
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm(image.convert("RGB")).unsqueeze(0)


def classify_mr(image_bytes: bytes) -> dict:
    """Note: Made synchronous because there are no await calls inside."""
    model, classes = _get_model()
    image = Image.open(io.BytesIO(image_bytes))
    tensor = _preprocess(image)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    top_label = classes[top_idx] if top_idx < len(classes) else "unknown"
    top_prob = float(probs[top_idx])

    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(classes[i], float(probs[i])) for i in top3_idx]

    result = {
        "disease_label": top_label,
        "confidence": top_prob * 100.0,
        "reasoning": "Local MR classifier inference.",
        "model_used": "mr_resnet",
        "raw_response": {"top3": top3},
        "all_probabilities": {classes[i]: float(probs[i]) * 100.0 for i in range(len(classes))},
        "raw_scores": probs.tolist()
    }

    logger.info("mr_prediction top_label=%s confidence=%.2f", top_label, top_prob)
    return result


def run_agent1(state: DiagnosticState) -> DiagnosticState:
    """
    Execute Agent 1: model inference using the mr_resnet.
    """
    session_id = state.get("session_id", "unknown")
    logger.info("[Agent 1] Starting for session %s", session_id)

    try:
        logger.info("[Agent 1] Running inference...")
        prediction = classify_mr(state["image_bytes"])

        state["raw_probabilities"] = prediction["all_probabilities"]
        state["predicted_class"]   = prediction["disease_label"]
        state["confidence_score"]  = prediction["confidence"]
        state["raw_scores"]        = prediction["raw_scores"]
        state["model_used"]        = prediction["model_used"]

        state["diagnosis_output"] = {
            "diagnosis_label": prediction["disease_label"],
            "confidence_pct": prediction["confidence"],
            "all_probabilities": prediction["all_probabilities"],
            "reasoning": prediction["reasoning"],
            "top3": prediction["raw_response"]["top3"]
        }
        state["clinical_notes"] = ""

        state["status"] = "agent1_complete"
        logger.info(
            "[Agent 1] Complete. Predicted: %s (%.2f%%)",
            prediction["disease_label"], prediction["confidence"]
        )

    except Exception as exc:
        logger.exception("[Agent 1] FAILED: %s", exc)
        state["status"] = "error"
        state["error"]  = f"Agent 1 failed: {exc}"

    return state
