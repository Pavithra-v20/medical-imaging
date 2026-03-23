from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import json
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError


# 🔹 Draw page border
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

    # Title
    elements.append(Paragraph("🩺 Patient Scan Report", styles['Title']))
    elements.append(Spacer(1, 20))

    # Table
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

    # Image (safe handling)
    image_path = data.get("image_path", "")

    if not os.path.isabs(image_path):
        image_path = os.path.join(os.getcwd(), image_path)

    if os.path.isfile(image_path):
        elements.append(Paragraph("Scan Image", heading_style))
        elements.append(Spacer(1, 10))
        elements.append(Image(image_path, width=4*inch, height=3*inch))
    else:
        elements.append(Paragraph("⚠️ Image not found", styles['Normal']))

    elements.append(Spacer(1, 25))

    # Summary
    elements.append(Paragraph("Summary", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(data["summary"], styles['Normal']))

    doc.build(elements, onFirstPage=draw_border, onLaterPages=draw_border)

    print("✅ PDF Generated")
    return pdf_path


# 🔹 Upload to Google Drive
def upload_to_drive(file_path):
    SCOPES = ['https://www.googleapis.com/auth/drive']

    try:
        creds = service_account.Credentials.from_service_account_file(
            'service_account.json', scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)

        file_metadata = {
            'name': os.path.basename(file_path),
            'parents':['1Tzb0NvuTsZPXNLNr3kiO5YoLL2FIXlBx']
        }

        media = MediaFileUpload(file_path, mimetype='application/pdf')

        try:
            # Try uploading to the specific folder
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
        except HttpError as e:
            if e.resp.status == 404:
                print(f"⚠️ Warning: Parent folder not found. Uploading to root instead.")
                # Fallback: remove 'parents' and upload to root
                del file_metadata['parents']
                file = service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id',
                    supportsAllDrives=True
                ).execute()
            else:
                raise e

        file_id = file.get('id')

        # make public
        service.permissions().create(
            fileId=file_id,
            body={'role': 'reader', 'type': 'anyone'}
        ).execute()

        return f"https://drive.google.com/file/d/{file_id}/view"
    
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return None


# 🔹 Main function
def main():
    # ✅ VALID JSON (no errors)
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

    # Parse JSON
    data = json.loads(data_json)

    # Generate PDF
    pdf_path = generate_report(data)

    # Upload to Drive
    drive_link = upload_to_drive(pdf_path)

    print("✅ FINAL OUTPUT:", drive_link)


if __name__ == "__main__":
    main()