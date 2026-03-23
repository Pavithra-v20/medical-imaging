"""
LangGraph state definition for the Medical AI Diagnostic System.

DiagnosticState is the shared TypedDict passed between all nodes in the
LangGraph StateGraph. Each agent reads what it needs and writes its own
output keys — other keys remain unchanged.

Key flow:
  Master Agent → Agent 1 → Agent 2 → Agent 3 → END
"""
from typing import TypedDict, Optional, Dict, Any, List


class DiagnosticState(TypedDict, total=False):
    """
    Shared state object threaded through all LangGraph nodes.

    Fields are marked Optional because the StateGraph starts with only
    the initial input fields populated; each agent adds its own.
    """

    # ── Inputs (set by the API layer before graph invocation) ────────────────
    session_id: str
    image_bytes: bytes          # raw uploaded image bytes
    image_path: str             # absolute path to saved image file
    patient_id: str
    technician_id: str
    physician_id: str
    modality: str               # 'mri' | 'xray' | 'ct'

    # ── Patient metadata (resolved from DB by the router) ─────────────────────
    patient_name: str
    patient_dob: str            # ISO date string
    patient_gender: str
    patient_contact: Optional[str]

    # ── Agent 1 — Medical Image Agent ────────────────────────────────────────
    # Step 1: model inference
    raw_probabilities: Dict[str, float]     # {class: probability%}
    predicted_class: str                    # top class name
    confidence_score: float                 # 0–100
    raw_scores: List[float]                 # raw softmax outputs

    # Step 2: Gemini LLM diagnosis
    diagnosis_output: Dict[str, Any]        # full Gemini JSON response
    clinical_notes: str                     # brief clinical commentary
    model_used: str                         # e.g. 'ViT-L16-fe + Xception'

    # ── Agent 2 — Visual Explanation Agent ──────────────────────────────────
    # Step A: Grad-CAM
    heatmap_path: str                       # local path to overlay PNG
    heatmap_b64: str                        # base64 PNG for API response

    # Step B: Gemini Vision clinical explanation
    explanation: str

    # ── Agent 3 — Report Agent ───────────────────────────────────────────────
    pdf_path: str
    report_link: str                        # Google Drive shareable URL

    # ── Control ──────────────────────────────────────────────────────────────
    status: str                             # 'pending' | 'complete' | 'error'
    error: Optional[str]                    # error message if status == 'error'
