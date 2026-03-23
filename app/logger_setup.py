import logging
import logging.config
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

def setup_logging():
    """
    Configures logging to route logs to categorized files.
    - Each main component (database, ingestion, etc.) gets its own log file.
    - A console handler shows all logs.
    - Uses JSON format for structured, machine-readable logs.
    """
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "level": "INFO",
            },
            "database_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": LOGS_DIR / "database.log",
                "maxBytes": 10485760, # 10MB
                "backupCount": 5,
            },
            "ingestion_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": LOGS_DIR / "ingestion.log",
                "maxBytes": 10485760,
                "backupCount": 5,
            },
            "prediction_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": LOGS_DIR / "prediction.log",
                "maxBytes": 10485760,
                "backupCount": 5,
            },
            "reporting_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": LOGS_DIR / "reporting.log",
                "maxBytes": 10485760,
                "backupCount": 5,
            },
            "segmentation_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": LOGS_DIR / "segmentation.log",
                "maxBytes": 10485760,
                "backupCount": 5,
            },
            "main_app_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": LOGS_DIR / "app.log",
                "maxBytes": 10485760,
                "backupCount": 5,
            },
        },
        "loggers": {
            "app.database": { "handlers": ["database_file", "console"], "level": "INFO", "propagate": False },
            "app.ingestion": { "handlers": ["ingestion_file", "console"], "level": "INFO", "propagate": False },
            "app.prediction": { "handlers": ["prediction_file", "console"], "level": "INFO", "propagate": False },
            "app.reporting": { "handlers": ["reporting_file", "console"], "level": "INFO", "propagate": False },
            "app.segmentation": { "handlers": ["segmentation_file", "console"], "level": "INFO", "propagate": False },
            "app": { "handlers": ["main_app_file", "console"], "level": "INFO", "propagate": False },
        },
        "root": {
            "handlers": ["console"],
            "level": "INFO",
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)
