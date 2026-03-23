"""
Model inference module for the Medical AI Diagnostic System.

Runs the ViT-L16-fe + Xception model on a preprocessed image tensor and
returns structured prediction results including the top class, its confidence
score, and a probability dictionary for all four tumour classes.
"""
import logging
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from backend.ml.model_loader import get_model, CLASS_NAMES
from backend.ml.preprocessor import preprocess_bytes

logger = logging.getLogger(__name__)


def run_inference(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run the hybrid model on raw image bytes and return structured predictions.

    Steps:
      1. Preprocess image bytes to a (1, 224, 224, 3) tensor.
      2. Call the PyTorch model to obtain softmax probabilities.
      3. Map probabilities to class names.
      4. Return the top class, confidence score, and full probability dict.

    Args:
        image_bytes: Raw bytes of the uploaded medical image.

    Returns:
        dict with keys:
          - ``predicted_class`` (str): highest-probability class name.
          - ``confidence`` (float): probability of the predicted class (0–100).
          - ``all_probabilities`` (dict[str, float]): per-class % probabilities.
          - ``raw_scores`` (list[float]): raw softmax outputs in CLASS_NAMES order.

    Raises:
        ValueError:   If image bytes cannot be decoded.
        RuntimeError: If inference fails.
    """
    model = get_model()

    # Preprocess -> torch tensor (N, C, H, W)
    tensor_np = preprocess_bytes(image_bytes)                 # (1, 224, 224, 3)
    tensor_torch = torch.from_numpy(tensor_np).permute(0, 3, 1, 2)  # (1, 3, 224, 224)

    logger.info("Running model inference …")
    try:
        model.eval()
        with torch.no_grad():
            logits = model(tensor_torch)
            probs = F.softmax(logits, dim=1)
            scores = probs[0].cpu().numpy()  # (4,)
    except Exception as exc:
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    # Map to class names
    all_probs: Dict[str, float] = {
        cls: float(round(float(score) * 100, 2))
        for cls, score in zip(CLASS_NAMES, scores)
    }

    top_idx: int = int(np.argmax(scores))
    predicted_class: str = CLASS_NAMES[top_idx]
    confidence: float = float(round(float(scores[top_idx]) * 100, 2))

    logger.info("Prediction: %s (%.2f%%)", predicted_class, confidence)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "raw_scores": [float(s) for s in scores],
    }
