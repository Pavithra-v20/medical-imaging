from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models

from app.config import get_settings
from app.utils.exceptions import PredictionError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _parse_classes(value: str) -> List[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def _load_state_dict(path: str) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise PredictionError("Unsupported model checkpoint format")

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
    if not settings.mr_model_path:
        raise PredictionError("MR_MODEL_PATH not set")

    classes = _parse_classes(settings.mr_classes)
    num_classes = len(classes)

    # Default to resnet18 unless the checkpoint requires a different arch
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state_dict = _load_state_dict(settings.mr_model_path)
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


async def classify_mr(image_path: str) -> dict:
    model, classes = _get_model()
    image = Image.open(image_path)
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
        "confidence": top_prob,
        "reasoning": "Local MR classifier inference.",
        "model_used": "mr_resnet",
        "raw_response": {"top3": top3},
    }

    logger.info("mr_prediction", top_label=top_label, confidence=top_prob)
    return result
