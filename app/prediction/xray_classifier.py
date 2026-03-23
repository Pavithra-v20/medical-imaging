import numpy as np
import torch
from PIL import Image

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return None
    if mask.shape[:2] == (size[1], size[0]):
        return mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize(size, resample=Image.NEAREST)
    return (np.array(mask_img) > 0).astype(np.uint8)


def _get_model():
    import torchxrayvision as xrv
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


def _preprocess(image: Image.Image, mask: np.ndarray | None):
    import torchxrayvision as xrv
    img = image.convert("L")
    arr = np.array(img, dtype=np.float32)

    if mask is not None:
        mask = _resize_mask(mask, img.size)
        arr = arr * mask

    arr = xrv.datasets.normalize(arr, 255)
    arr = arr[None, ...]
    arr = xrv.datasets.XRayCenterCrop()(arr)
    arr = xrv.datasets.XRayResizer(224)(arr)

    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


async def classify_xray(image_path: str, mask: np.ndarray | None = None) -> dict:
    image = Image.open(image_path)
    model = _get_model()
    tensor = _preprocess(image, mask)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    pathologies = list(model.pathologies)
    top_idx = int(np.argmax(probs))
    top_label = pathologies[top_idx]
    top_prob = float(probs[top_idx])

    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(pathologies[i], float(probs[i])) for i in top3_idx]
    reasoning = "Top findings: " + ", ".join([f"{k} ({v*100:.1f}%)" for k, v in top3])

    result = {
        "disease_label": top_label,
        "confidence": top_prob,
        "reasoning": reasoning,
        "model_used": "xray_densenet121_res224_all",
        "raw_response": {"top3": top3},
    }

    logger.info("xray_prediction", top_label=top_label, confidence=top_prob)
    return result
