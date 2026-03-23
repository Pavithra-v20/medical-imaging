"""
Google Drive uploader using a Service Account (no OAuth required).

The `service_account.json` at the project root is used directly — no
browser-based token refresh is ever needed. The uploaded PDF is made
publicly readable via a `reader` permission for `type=anyone`.

Usage:
    link = upload_to_drive("/path/to/report_abc123.pdf", "abc123")
"""
import logging
import os

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from backend.config import settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _get_service():
    """
    Build and return an authenticated Google Drive API service client.

    Uses the service account JSON file specified in settings.
    The service account must have been granted 'Editor' access to the
    target Drive folder.

    Returns:
        googleapiclient.discovery.Resource: Authenticated Drive v3 service.

    Raises:
        FileNotFoundError: If `service_account.json` does not exist.
        google.auth.exceptions.TransportError: If credentials are invalid.
    """
    sa_path = settings.DRIVE_SERVICE_ACCOUNT_JSON
    if not os.path.exists(sa_path):
        raise FileNotFoundError(f"Service account file not found: {sa_path}")

    creds = Credentials.from_service_account_file(sa_path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def upload_to_drive(file_path: str, session_id: str) -> str:
    """
    Upload a PDF file to Google Drive and return a shareable link.

    Steps:
      1. Authenticate via the service account credentials.
      2. Upload the file to the configured Drive folder.
      3. Set public reader permission (`type=anyone`, `role=reader`).
      4. Return the standard `/view` shareable URL.

    Args:
        file_path:  Absolute local path to the PDF file to upload.
        session_id: Used to name the file in Drive (prefix for easy lookup).

    Returns:
        str: Google Drive shareable URL in the form
             `https://drive.google.com/file/d/{file_id}/view`.

    Raises:
        FileNotFoundError: If the local PDF file does not exist.
        RuntimeError:      If the upload or permission API call fails.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    service = _get_service()

    drive_filename = f"MedAI_Report_{session_id[:8].upper()}.pdf"

    file_metadata = {
        "name":    drive_filename,
        "parents": [settings.DRIVE_FOLDER_ID],
    }

    media = MediaFileUpload(file_path, mimetype="application/pdf", resumable=False)

    logger.info("Uploading %s to Drive as '%s' …", file_path, drive_filename)
    try:
        uploaded = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id",
        ).execute()
    except Exception as exc:
        raise RuntimeError(f"Google Drive upload failed: {exc}") from exc

    file_id = uploaded["id"]

    # Make publicly accessible (view only)
    try:
        service.permissions().create(
            fileId=file_id,
            body={"role": "reader", "type": "anyone"},
        ).execute()
    except Exception as exc:
        logger.warning("Could not set public permission on Drive file: %s", exc)

    link = f"https://drive.google.com/file/d/{file_id}/view"
    logger.info("Drive upload complete. Link: %s", link)
    return link
