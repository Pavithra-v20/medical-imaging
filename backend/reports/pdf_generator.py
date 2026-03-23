"""
PDF Report Generator using ReportLab.
=======================================
Produces a comprehensive, professionally formatted diagnostic report PDF
from the final DiagnosticState that includes:

  - Hospital header with logo placeholder and report metadata
  - Patient demographics table
  - Diagnosis result table with confidence score
  - Side-by-side images: Original scan | Grad-CAM heatmap overlay
  - Clinical explanation (Gemini Vision output)
  - Per-class probability breakdown
  - Physician review section (blank, to be filled manually)
  - Page border and footer
"""
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import BalancedColumns

from backend.agents.state import DiagnosticState
from backend.config import settings

logger = logging.getLogger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#0D1B2A")
MED_BLUE    = colors.HexColor("#1E3A5F")
ACCENT_TEAL = colors.HexColor("#00B4D8")
LIGHT_GREY  = colors.HexColor("#F5F7FA")
MID_GREY    = colors.HexColor("#CBD5E0")
TEXT_DARK   = colors.HexColor("#1A202C")
SUCCESS_GRN = colors.HexColor("#38A169")
WARN_AMBER  = colors.HexColor("#D69E2E")
DANGER_RED  = colors.HexColor("#E53E3E")

SEVERITY_COLORS = {
    "Normal":        SUCCESS_GRN,
    "Benign":        WARN_AMBER,
    "Malignant":     DANGER_RED,
    "Indeterminate": MID_GREY,
}


def _draw_page_border(canvas, doc):
    """Draw a double-line border and footer on every page."""
    w, h = doc.pagesize
    margin = 15

    # Outer border
    canvas.setStrokeColor(MED_BLUE)
    canvas.setLineWidth(2)
    canvas.rect(margin, margin, w - 2 * margin, h - 2 * margin)

    # Inner border (thin)
    canvas.setStrokeColor(ACCENT_TEAL)
    canvas.setLineWidth(0.5)
    canvas.rect(margin + 4, margin + 4, w - 2 * (margin + 4), h - 2 * (margin + 4))

    # Footer
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MID_GREY)
    canvas.drawCentredString(
        w / 2, margin + 6,
        f"Medical AI Diagnostic System  ·  Confidential  ·  Page {doc.page}"
    )


def _styles() -> Dict[str, ParagraphStyle]:
    """Create and return all custom Paragraph styles."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=20,
            textColor=DARK_BLUE,
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=10,
            textColor=ACCENT_TEAL,
            spaceAfter=6,
            alignment=TA_CENTER,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader",
            parent=base["Heading2"],
            fontSize=11,
            textColor=MED_BLUE,
            spaceBefore=10,
            spaceAfter=4,
            borderPad=3,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=9,
            textColor=TEXT_DARK,
            leading=14,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["Normal"],
            fontSize=8,
            textColor=MID_GREY,
        ),
        "diagnosis": ParagraphStyle(
            "DiagnosisLabel",
            parent=base["Normal"],
            fontSize=14,
            textColor=DARK_BLUE,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
        ),
    }


def build_report(state: DiagnosticState) -> str:
    """
    Build a formatted ReportLab PDF from the final DiagnosticState.

    Args:
        state: Completed DiagnosticState containing all agent outputs.

    Returns:
        str: Absolute path to the saved PDF file.

    Raises:
        OSError: If the output directory cannot be created.
        Exception: If ReportLab fails to build the document.
    """
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)

    session_id = state.get("session_id", "unknown")
    filename   = f"report_{session_id}.pdf"
    pdf_path   = os.path.join(settings.REPORTS_DIR, filename)

    styles = _styles()
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    elements = []

    # ── Header ────────────────────────────────────────────────────────────────
    elements.append(Paragraph("MEDICAL AI DIAGNOSTIC REPORT", styles["title"]))
    elements.append(Paragraph("Powered by ViT-L16-fe + Xception · Grad-CAM Explainability",
                               styles["subtitle"]))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_TEAL, spaceAfter=6))

    # Report meta
    meta_data = [
        ["Session ID", session_id[:8].upper() + "…",
         "Date", datetime.now().strftime("%d %b %Y %H:%M")],
        ["Patient ID", state.get("patient_id", "—")[:8],
         "Modality", state.get("modality", "MRI").upper()],
        ["Model", state.get("model_used", "ViT-L16-fe + Xception"),
         "Status", "Complete"],
    ]
    meta_table = Table(meta_data, colWidths=[2.5 * cm, 5 * cm, 2.5 * cm, 5 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), MED_BLUE),
        ("TEXTCOLOR",   (2, 0), (2, -1), MED_BLUE),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, MID_GREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # ── Patient Demographics ────────────────────────────────────────────────
    elements.append(Paragraph("PATIENT INFORMATION", styles["section_header"]))
    pat_data = [
        ["Name", state.get("patient_name", "—"), "DOB", state.get("patient_dob", "—")],
        ["Gender", state.get("patient_gender", "—"), "Contact", state.get("patient_contact", "—")],
    ]
    pat_table = Table(pat_data, colWidths=[2.5 * cm, 7 * cm, 2.5 * cm, 3 * cm])
    pat_table.setStyle(TableStyle([
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), MED_BLUE),
        ("TEXTCOLOR",   (2, 0), (2, -1), MED_BLUE),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, MID_GREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(pat_table)
    elements.append(Spacer(1, 10))

    # ── Diagnosis Result ──────────────────────────────────────────────────────
    elements.append(Paragraph("DIAGNOSIS RESULT", styles["section_header"]))

    dout: dict = state.get("diagnosis_output", {})
    diagnosis_label  = dout.get("diagnosis_label", state.get("predicted_class", "Unknown")).replace("_", " ").title()
    confidence_label = dout.get("confidence_label", "—")
    severity         = dout.get("severity", "Indeterminate")
    conf_score       = state.get("confidence_score", 0.0)
    recommended_act  = dout.get("recommended_action", "—")
    differential     = ", ".join(dout.get("differential_diagnoses", []))
    sev_color        = SEVERITY_COLORS.get(severity, MID_GREY)

    diag_data = [
        ["Diagnosis",    diagnosis_label,   "Severity",      severity],
        ["Confidence",   f"{conf_score:.2f}%", "Confidence Level", confidence_label],
        ["Recommended Action", recommended_act, "Differential", differential or "—"],
    ]
    diag_table = Table(diag_data, colWidths=[3 * cm, 5.5 * cm, 3 * cm, 5.5 * cm])
    diag_table.setStyle(TableStyle([
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), MED_BLUE),
        ("TEXTCOLOR",   (2, 0), (2, -1), MED_BLUE),
        ("BACKGROUND",  (1, 0), (1, 0), LIGHT_GREY),
        ("TEXTCOLOR",   (1, 1), (1, 1), sev_color),
        ("FONTNAME",    (1, 1), (1, 1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GREY, colors.white, LIGHT_GREY]),
        ("GRID",        (0, 0), (-1, -1), 0.5, MID_GREY),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("SPAN",        (1, 2), (1, 2)),
    ]))
    elements.append(diag_table)
    elements.append(Spacer(1, 5))

    # Clinical note
    clinical_note = dout.get("clinical_note", state.get("clinical_notes", ""))
    if clinical_note:
        elements.append(Paragraph(f"<b>Clinical Note:</b> {clinical_note}", styles["body"]))
    elements.append(Spacer(1, 10))

    # ── Probability Breakdown ────────────────────────────────────────────────
    elements.append(Paragraph("MODEL CONFIDENCE SCORES", styles["section_header"]))
    raw_probs: dict = state.get("raw_probabilities", {})
    if raw_probs:
        prob_data = [["Class", "Confidence (%)"]]
        for cls, prob in sorted(raw_probs.items(), key=lambda x: -x[1]):
            prob_data.append([cls.replace("_", " ").title(), f"{prob:.2f}%"])
        prob_table = Table(prob_data, colWidths=[8 * cm, 7 * cm])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), MED_BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.5, MID_GREY),
            ("ALIGN",       (1, 0), (1, -1), "RIGHT"),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(prob_table)
    elements.append(Spacer(1, 10))

    # ── Scan & Heatmap Images ─────────────────────────────────────────────────
    elements.append(Paragraph("SCAN ANALYSIS — ORIGINAL vs GRAD-CAM HEATMAP", styles["section_header"]))

    image_path   = state.get("image_path", "")
    heatmap_path = state.get("heatmap_path", "")

    img_elements = []
    if image_path and os.path.isfile(image_path):
        orig_img = RLImage(image_path, width=7.5 * cm, height=6.5 * cm)
        img_elements.append([orig_img, Paragraph("Original Scan", styles["small"])])
    if heatmap_path and os.path.isfile(heatmap_path):
        heat_img = RLImage(heatmap_path, width=7.5 * cm, height=6.5 * cm)
        img_elements.append([heat_img, Paragraph("Grad-CAM Heatmap", styles["small"])])

    if img_elements:
        if len(img_elements) == 2:
            img_table_data = [
                [img_elements[0][0], img_elements[1][0]],
                [img_elements[0][1], img_elements[1][1]],
            ]
        else:
            img_table_data = [[img_elements[0][0]], [img_elements[0][1]]]
        img_table = Table(img_table_data, colWidths=[8 * cm, 8 * cm],
                          hAlign="CENTER")
        img_table.setStyle(TableStyle([
            ("ALIGN",    (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",     (0, 0), (-1, -1), 0.5, MID_GREY),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(img_table)
    else:
        elements.append(Paragraph("Images not available.", styles["body"]))
    elements.append(Spacer(1, 10))

    # ── Clinical Explanation ──────────────────────────────────────────────────
    explanation = state.get("explanation", "")
    if explanation:
        elements.append(Paragraph("VISUAL EXPLANATION — CLINICAL INTERPRETATION", styles["section_header"]))
        for line in explanation.split("\n"):
            if line.strip():
                elements.append(Paragraph(line.strip(), styles["body"]))
                elements.append(Spacer(1, 3))
        elements.append(Spacer(1, 10))

    # ── Physician Review Section ──────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=1, color=MID_GREY, spaceAfter=6))
    elements.append(Paragraph("PHYSICIAN REVIEW", styles["section_header"]))
    elements.append(Paragraph(
        "This AI-generated report is for clinical assistance only. "
        "Final diagnosis must be confirmed by a licensed physician.",
        styles["small"]
    ))
    elements.append(Spacer(1, 8))

    review_rows = [
        ["Reviewing Physician:", "___________________________________", "Date:", "_________________"],
        ["Signature:",           "___________________________________", "Stamp:", ""],
        ["Review Notes:", "", "", ""],
        ["", "", "", ""],
        ["", "", "", ""],
    ]
    rev_table = Table(review_rows, colWidths=[3.5 * cm, 7.5 * cm, 2 * cm, 4 * cm])
    rev_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, 1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (1, 0), (1, 1), 0.5, colors.black),
        ("LINEBELOW", (3, 0), (3, 0), 0.5, colors.black),
        ("SPAN",     (1, 2), (-1, 4)),
        ("BOX",      (1, 2), (-1, 4), 0.5, colors.black),
    ]))
    elements.append(rev_table)

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(
        elements,
        onFirstPage=_draw_page_border,
        onLaterPages=_draw_page_border,
    )
    logger.info("PDF report built: %s", pdf_path)
    return pdf_path
