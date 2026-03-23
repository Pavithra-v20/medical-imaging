"""
Configuration module for the Medical AI Diagnostic System.
Loads all environment variables and exposes typed settings.
"""
from pydantic_settings import BaseSettings
from pathlib import Path

# Root of the entire project (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/medical_db"

    # Gemini
    GEMINI_API_KEY: str = ""

    # Google Drive
    DRIVE_SERVICE_ACCOUNT_JSON: str = str(PROJECT_ROOT / "service_account.json")
    DRIVE_FOLDER_ID: str = "1Tzb0NvuTsZPXNLNr3kiO5YoLL2FIXlBx"

    # Model (PyTorch)
    MODEL_PT_PATH: str = str(PROJECT_ROOT / "Model" / "best_mr (3).pt")

    # File storage
    UPLOAD_DIR: str = str(PROJECT_ROOT / "uploads")
    HEATMAP_DIR: str = str(PROJECT_ROOT / "heatmaps")
    REPORTS_DIR: str = str(PROJECT_ROOT / "reports_output")

    # JWT Auth (simple)
    SECRET_KEY: str = "medical-ai-secret-key-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"


settings = Settings()
