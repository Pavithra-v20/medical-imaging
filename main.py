from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import json
import os

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = ['https://www.googleapis.com/auth/drive.file']


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


# 🔹 Get OAuth credentials
def get_credentials():
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds


# 🔹 Upload to Google Drive
def upload_to_drive(file_path):
    creds = get_credentials()
    service = build('drive', 'v3', credentials=creds)

    FOLDER_ID = '1Tzb0NvuTsZPXNLNr3kiO5YoLL2FIXlBx'  # your Reports folder ID

    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [FOLDER_ID]
    }

    media = MediaFileUpload(file_path, mimetype='application/pdf')

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    file_id = file.get('id')

    # Make file publicly readable
    service.permissions().create(
        fileId=file_id,
        body={'role': 'reader', 'type': 'anyone'}
    ).execute()

    link = f"https://drive.google.com/file/d/{file_id}/view"
    print("📤 Uploaded to Drive")
    print("🔗 Link:", link)
    return link


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
    drive_link = upload_to_drive(pdf_path)
    print("✅ FINAL OUTPUT:", drive_link)


if __name__ == "__main__":
    main()