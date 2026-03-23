"""
PyTorch model loader for the brain tumour classifier (.pt checkpoint).

The new model is provided as `Model/best_mr (3).pt`. We load it once,
cache the torch.nn.Module, and serve it for inference.
"""
import logging
import os
from typing import Optional

import torch

from backend.config import settings

logger = logging.getLogger(__name__)

# Output classes in the order the model's softmax layer produces them
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

_model: Optional[torch.nn.Module] = None
_device = torch.device("cpu")


def _load_pt_model(path: str) -> torch.nn.Module:
    """Load a PyTorch .pt checkpoint (either full module or scripted graph)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    logger.info("Loading PyTorch model from %s …", path)
    obj = torch.load(path, map_location=_device)

    # If the checkpoint is already a Module/JIT module, return directly.
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    # TorchScript module (torch.jit.ScriptModule / TracedModule)
    if hasattr(obj, "forward"):
        obj.eval()
        return obj

    raise RuntimeError(
        "Unsupported checkpoint format. Expected a saved torch.nn.Module or TorchScript object."
    )


def load_model() -> torch.nn.Module:
    """Return the cached PyTorch model, loading it if necessary."""
    global _model
    if _model is not None:
        return _model

    model_path = settings.MODEL_PT_PATH
    _model = _load_pt_model(model_path)
    logger.info("Model loaded. Ready for inference on %s.", _device)
    return _model


def get_model() -> torch.nn.Module:
    """Public accessor used by predictors."""
    return load_model()
