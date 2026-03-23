from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = ""
    database_enabled: bool = True

    # Prediction models
    prediction_model: str = "mr"  # "local_rules" or "mr"

    # Local MR classifier
    mr_model_path: str = ""
    mr_classes: str = "glioma,meningioma,notumor,pituitary"

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
    prediction_enabled: bool = True

    # LLM explanation (optional)
    explanation_enabled: bool = False
    nim_api_key: str = ""
    nim_model: str = "meta/llama-3.1-8b-instruct"
    nim_base_url: str = "https://integrate.api.nvidia.com/v1/"


    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
