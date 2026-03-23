from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from urllib.parse import quote
import json
import os
import requests


UPLOADCARE_PUBLIC_KEY = "d3ed92046c50ac1e2698"


# 🔹 Draw border
def draw_border(canvas, doc):
    width, height = doc.pagesize
    margin = 20
    canvas.setStrokeColor(colors.grey)
    canvas.setLineWidth(2)
    canvas.rect(margin, margin, width - 2*margin, height - 2*margin)


# 🔹 Generate PDF
def generate_report(data):
    pdf_path = "patient_report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        textColor=colors.darkblue
    )

    elements = []
    elements.append(Paragraph("Patient Scan Report", styles['Title']))
    elements.append(Spacer(1, 20))

    table_data = [
        ["Field", "Details"],
        ["Patient ID", data["patient_id"]],
        ["Name", data["name"]],
        ["Scan Type", data["scan_type"]],
        ["Result", data["result"]]
    ]

    table = Table(table_data, colWidths=[2.5*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])
    ]))

    elements.append(table)
    elements.append(Spacer(1, 25))

    image_path = data.get("image_path", "")
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.getcwd(), image_path)

    if os.path.isfile(image_path):
        elements.append(Paragraph("Scan Image", heading_style))
        elements.append(Spacer(1, 10))
        elements.append(Image(image_path, width=4*inch, height=3*inch))
    else:
        elements.append(Paragraph("Image not found", styles['Normal']))

    elements.append(Spacer(1, 25))
    elements.append(Paragraph("Summary", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(data["summary"], styles['Normal']))

    doc.build(elements, onFirstPage=draw_border, onLaterPages=draw_border)
    print("✅ PDF Generated")
    return pdf_path


# 🔹 Upload to Uploadcare
def upload_to_cloud(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            'https://upload.uploadcare.com/base/',
            data={
                'UPLOADCARE_PUB_KEY': UPLOADCARE_PUBLIC_KEY,
                'UPLOADCARE_STORE':   '1',
            },
            files={
                'file': (os.path.basename(file_path), f, 'application/pdf')
            }
        )

    result   = response.json()
    file_id  = result['file']
    filename = os.path.basename(file_path)

    # Raw file URL
    raw_url = f"https://ucarecdn.com/{file_id}/{filename}"

    # Wrap in Mozilla PDF.js viewer — opens PDF in browser guaranteed
    viewer_url = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={quote(raw_url, safe='')}"

    print("📤 Uploaded to Uploadcare")
    print("🔗 View Link:", viewer_url)
    return viewer_url


# 🔹 Main
def main():
    data_json = """
    {
        "patient_id": "P12345",
        "name": "Surya",
        "scan_type": "MRI Brain",
        "result": "No abnormality detected",
        "summary": "The MRI scan shows normal brain structure with no issues.",
        "image_path": "Data/Head-MRI.jpg"
    }
    """

    data = json.loads(data_json)
    pdf_path = generate_report(data)
    cloud_link = upload_to_cloud(pdf_path)
    print("✅ FINAL OUTPUT:", cloud_link)


if __name__ == "__main__":
    main()