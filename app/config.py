from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str

    # NVIDIA NIM
    vista3d_nim_url: str
    nvidia_api_key: str

    # Triton
    triton_url: str = "localhost:8000"

    # Prediction models
    ctchat_url: str
    medgemma_url: str
    google_api_key: str = ""
    prediction_model: str = "ctchat"  # "ctchat" or "medgemma"

    # Review rules
    confidence_threshold: float = 0.75

    # Storage
    storage_uploads: str = "storage/uploads"
    storage_processed: str = "storage/processed"
    storage_masks: str = "storage/masks"
    storage_reports: str = "storage/reports"

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
