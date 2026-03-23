"""
Agent 2 — Visual Explanation Agent
====================================
Responsible for making the diagnosis visually interpretable and clinically
explainable via two sequential internal steps:

  Step A — Visual Explanation (Grad-CAM):
    Hooks into the Xception branch of the model that is already loaded in
    memory from Agent 1. Generates a Grad-CAM heatmap overlay highlighting
    the regions most influential in the prediction. Saves the overlay PNG.

  Step B — Clinical Explanation (Gemini Vision):
    Sends the original scan + heatmap overlay + Agent 1 diagnosis to Gemini
    Vision. Receives a structured clinical explanation covering anatomical
    regions, confidence trust, and clinical reasoning.

Both steps must complete before Agent 2 signals completion to the Master.
"""
import base64
import logging
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from backend.agents.state import DiagnosticState
from backend.config import settings
from backend.ml.gradcam import generate_heatmap

logger = logging.getLogger(__name__)

# Shared Gemini client
_gemini_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a cached Gemini API client."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def _build_explanation_prompt(
    predicted_class: str,
    confidence: float,
    diagnosis_output: dict,
) -> str:
    """
    Build a Gemini Vision prompt for clinical explanation of the heatmap.

    Args:
        predicted_class:  Model's top predicted class name.
        confidence:       Confidence score percentage.
        diagnosis_output: Structured diagnosis JSON from Agent 1.

    Returns:
        str: Multi-part prompt string.
    """
    label = diagnosis_output.get("diagnosis_label", predicted_class)
    severity = diagnosis_output.get("severity", "Unknown")
    clinical_note = diagnosis_output.get("clinical_note", "")

    return f"""You are a neuroradiologist AI reviewing a brain MRI scan with a Grad-CAM heatmap overlay.

The deep learning model predicted: **{label}** with {confidence:.1f}% confidence.
Severity classification: {severity}
Clinical context: {clinical_note}

The first image is the ORIGINAL brain MRI scan.
The second image is the GRAD-CAM HEATMAP OVERLAY — hot (red/yellow) regions indicate
areas that most influenced the model's prediction.

Provide a structured clinical explanation in plain language covering:
1. HIGHLIGHTED REGIONS: Which anatomical structures are highlighted in the heatmap?
2. DIAGNOSTIC REASONING: Why do those regions contribute to the predicted diagnosis?
3. CONFIDENCE ASSESSMENT: Given the heatmap focus and confidence score, how reliable is this prediction?
4. CLINICAL RECOMMENDATION: What should the radiologist prioritise when reviewing this case?

Keep the response concise and clinically meaningful (200-300 words).
Format as numbered sections matching the headers above."""


def run_agent2(state: DiagnosticState) -> DiagnosticState:
    """
    Execute Agent 2: Grad-CAM heatmap generation + Gemini Vision explanation.

    Only runs if state["status"] == "agent1_complete".

    Reads:
        state["image_bytes"]       — uploaded image for Grad-CAM
        state["session_id"]        — used as heatmap filename
        state["predicted_class"]   — for Grad-CAM class targeting
        state["confidence_score"]  — for explanation prompt
        state["diagnosis_output"]  — from Agent 1 Gemini

    Writes:
        state["heatmap_path"]  — abs path to saved overlay PNG
        state["heatmap_b64"]   — base64 PNG string
        state["explanation"]   — Gemini Vision clinical explanation text

    Args:
        state: Current DiagnosticState (must have Agent 1 outputs).

    Returns:
        Updated DiagnosticState with Agent 2 fields populated.
    """
    session_id = state.get("session_id", "unknown")
    logger.info("[Agent 2] Starting for session %s", session_id)

    # Guard: only run if Agent 1 succeeded
    if state.get("status") == "error":
        logger.warning("[Agent 2] Skipping — upstream error detected.")
        return state

    try:
        # ── Step A: Generate Grad-CAM heatmap ─────────────────────────────────
        logger.info("[Agent 2] Step A — Generating Grad-CAM heatmap …")

        heatmap_path, heatmap_b64 = generate_heatmap(
            image_bytes=state["image_bytes"],
            session_id=session_id,
        )
        state["heatmap_path"] = heatmap_path
        state["heatmap_b64"]  = heatmap_b64

        logger.info("[Agent 2] Step A complete. Heatmap: %s", heatmap_path)

        # ── Step B: Gemini Vision clinical explanation ─────────────────────────
        logger.info("[Agent 2] Step B — Calling Gemini Vision for explanation …")

        # Gemini client
        client = _get_client()

        prompt = _build_explanation_prompt(
            predicted_class=state.get("predicted_class", "unknown"),
            confidence=state.get("confidence_score", 0.0),
            diagnosis_output=state.get("diagnosis_output", {}),
        )

        # Build multimodal content: text prompt + original scan + heatmap overlay
        content_parts = [
            prompt,
            genai_types.Part.from_bytes(
                data=state["image_bytes"],
                mime_type="image/jpeg",
            ),
            genai_types.Part.from_bytes(
                data=base64.b64decode(heatmap_b64),
                mime_type="image/png",
            ),
        ]

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content_parts,
        )
        state["explanation"] = response.text.strip()

        logger.info("[Agent 2] Step B complete. Explanation length: %d chars",
                    len(state["explanation"]))

        state["status"] = "agent2_complete"

    except Exception as exc:
        logger.exception("[Agent 2] FAILED: %s", exc)
        state["status"] = "error"
        state["error"]  = f"Agent 2 failed: {exc}"

    return state
