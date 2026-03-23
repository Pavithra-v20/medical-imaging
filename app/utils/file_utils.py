import os
import shutil
from pathlib import Path
from fastapi import UploadFile
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def save_upload(file: UploadFile, upload_dir: str, session_id: str) -> Path:
    """Save an uploaded DICOM file to disk and return its path."""
    dest_dir = Path(upload_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix or ".dcm"
    dest_path = dest_dir / f"{session_id}{suffix}"

    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info("file_saved", path=str(dest_path), size_bytes=len(content))
    return dest_path


def cleanup_temp_files(paths: list):
    """Delete temporary files created during processing."""
    for path in paths:
        if path and Path(path).exists():
            try:
                os.remove(path)
                logger.info("temp_file_removed", path=str(path))
            except Exception as e:
                logger.warning("cleanup_failed", path=str(path), error=str(e))


def ensure_dirs(*dirs: str):
    """Create directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
