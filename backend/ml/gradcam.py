"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the hybrid model.

The trained model has two branches:
  - ViT-L16-fe branch  (transformer — no conv layers, cannot be used for Grad-CAM)
  - Xception CNN branch (named 'xception' layer containing sub-layers)

Strategy:
  1. Find the Xception sub-model inside the full model.
  2. Target its last separable-conv activation  (`block14_sepconv2_act`).
  3. Build a two-output sub-model:  [conv_activations, final_predictions].
  4. Use tf.GradientTape to compute gradients of the top-class score
     w.r.t. the last conv activations.
  5. Pool gradients spatially → weight activation maps → ReLU → normalise.
  6. Resize heatmap to original image dimensions and overlay with colour map.
"""
import logging
import os
import uuid
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from backend.config import settings
from backend.ml.model_loader import get_model
from backend.ml.preprocessor import preprocess_bytes, tensor_to_pil

logger = logging.getLogger(__name__)

# The last convolutional activation in the Xception branch
XCEPTION_LAST_CONV = "block14_sepconv2_act"


def _find_xception_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """
    Locate the Xception sub-model or the target conv layer in the full model.

    Searches recursively through nested layers to find the layer named
    `block14_sepconv2_act` — the final spatial activation in Xception.

    Args:
        model: The full hybrid ViT + Xception Keras model.

    Returns:
        The target Keras layer.

    Raises:
        ValueError: If the Xception last-conv layer cannot be found.
    """
    for layer in model.layers:
        # Check if this layer IS the target conv
        if layer.name == XCEPTION_LAST_CONV:
            return layer
        # Recurse into nested models (e.g. the Xception sub-model)
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                if sub_layer.name == XCEPTION_LAST_CONV:
                    return sub_layer

    # Fallback: use the last Conv2D-like layer in the model
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.SeparableConv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            logger.warning("Using fallback conv layer: %s", layer.name)
            return layer

    raise ValueError(
        f"Cannot find Grad-CAM target layer '{XCEPTION_LAST_CONV}' "
        "or any Conv2D-type layer in the model."
    )


def generate_heatmap(
    image_bytes: bytes,
    session_id: str,
    class_index: int = None,
) -> Tuple[str, str]:
    """
    Generate a Grad-CAM heatmap overlay and save it to disk.

    Args:
        image_bytes: Raw bytes of the uploaded medical image.
        session_id:  Unique session identifier used as the output filename.
        class_index: Class index to compute Grad-CAM for. If None, uses the
                     model's predicted (argmax) class.

    Returns:
        tuple:
          - heatmap_path (str): Absolute path to the saved overlay PNG.
          - heatmap_b64  (str): Base64-encoded overlay PNG for API responses.

    Raises:
        RuntimeError: If gradient computation or image saving fails.
    """
    model = get_model()

    # ── 1. Find target conv layer ────────────────────────────────────────────
    conv_layer = _find_xception_layer(model)

    # ── 2. Build grad model with two outputs ─────────────────────────────────
    # Output 1: last conv activations  (H, W, C)
    # Output 2: final model predictions (num_classes,)
    try:
        # For nested Xception layer, we need the internal sub-model
        xception_submodel = None
        for lyr in model.layers:
            if hasattr(lyr, "layers") and any(
                sl.name == XCEPTION_LAST_CONV for sl in lyr.layers
            ):
                xception_submodel = lyr
                break

        if xception_submodel is not None:
            conv_output = conv_layer.output
            # Create a model from the main input → conv_output + final predictions
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[conv_layer.output, model.output]
            )
        else:
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[conv_layer.output, model.output]
            )
    except Exception as exc:
        logger.warning("Standard grad_model build failed (%s). Using fallback.", exc)
        # Fallback: use last conv-like layer directly in the full model
        last_conv = None
        for lyr in model.layers:
            if hasattr(lyr, "output_shape") and len(getattr(lyr, "output_shape", ())) == 4:
                last_conv = lyr
        if last_conv is None:
            raise RuntimeError("No suitable conv layer found for Grad-CAM.")
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )

    # ── 3. Preprocess image ──────────────────────────────────────────────────
    tensor = tf.constant(preprocess_bytes(image_bytes), dtype=tf.float32)

    # ── 4. Compute gradients ─────────────────────────────────────────────────
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        conv_outputs, predictions = grad_model(tensor, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)    # (1, H, W, C)

    # ── 5. Pool gradients and weight activation maps ──────────────────────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
    conv_outputs = conv_outputs[0]                         # (H, W, C)

    # Weighted combination of feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                            # (H, W)
    heatmap = tf.nn.relu(heatmap)                           # ReLU
    heatmap_np: np.ndarray = heatmap.numpy()

    # Normalise to [0, 1]
    if heatmap_np.max() > 0:
        heatmap_np = heatmap_np / heatmap_np.max()

    # ── 6. Resize and overlay on original image ───────────────────────────────
    original_pil = tensor_to_pil(preprocess_bytes(image_bytes))
    orig_w, orig_h = original_pil.size              # PIL: (width, height)

    heatmap_resized = cv2.resize(
        (heatmap_np * 255).astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # BGR
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original_np = np.array(original_pil)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap_rgb, 0.4, 0)

    # ── 7. Save overlay to disk ───────────────────────────────────────────────
    os.makedirs(settings.HEATMAP_DIR, exist_ok=True)
    filename = f"heatmap_{session_id}.png"
    heatmap_path = os.path.join(settings.HEATMAP_DIR, filename)

    overlay_pil = Image.fromarray(overlay.astype(np.uint8))
    overlay_pil.save(heatmap_path)
    logger.info("Heatmap saved to %s", heatmap_path)

    # ── 8. Base64 encode for API ──────────────────────────────────────────────
    import base64, io
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return heatmap_path, heatmap_b64
