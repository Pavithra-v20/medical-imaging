"""
FastAPI Application Entry Point — Medical AI Diagnostic System
===============================================================
Starts the FastAPI app, registers all routers, handles CORS, loads the
ML model at startup, and initialises the database tables.

Start with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database.db import init_db
from backend.agents.agent1_medical_image import _get_model as load_model
from backend.routers import sessions, patients, physicians, technicians

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    On startup:
      1. Creates required directories (uploads, heatmaps, reports_output).
      2. Initialises all database tables via SQLAlchemy.
      3. Pre-loads the ViT-L16-fe + Xception model into memory so the
         first request is not delayed by model loading.

    On shutdown:
      Logs a clean shutdown message.
    """
    # Create storage directories
    for d in [settings.UPLOAD_DIR, settings.HEATMAP_DIR, settings.REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)
        logger.info("Directory ready: %s", d)

    # Initialise DB tables (non-fatal — server still starts if DB is unreachable)
    logger.info("Initialising database tables …")
    try:
        init_db()
        logger.info("Database tables ready.")
    except Exception as exc:
        logger.error("Database init failed: %s", exc)
        logger.warning(
            "\n"
            "  ┌─────────────────────────────────────────────────────────────┐\n"
            "  │  DATABASE NOT CONNECTED — check your .env DATABASE_URL      │\n"
            "  │  Common fix: set the correct postgres password, e.g.:       │\n"
            "  │    DATABASE_URL=postgresql://postgres:PASSWORD@localhost/... │\n"
            "  │  Find the password in pgAdmin → right-click server →        │\n"
            "  │  Properties → Connection tab (or check installer notes).    │\n"
            "  └─────────────────────────────────────────────────────────────┘"
        )
        logger.warning("API server is running but DB endpoints will fail until fixed.")

    # Pre-load model
    logger.info("Pre-loading AI model … (this may take a moment)")
    try:
        load_model()
        logger.info("Model loaded and cached.")
    except Exception as exc:
        logger.error("Model load failed at startup: %s", exc)
        logger.warning("System will attempt model load on first request.")

    yield  # App is running

    logger.info("Medical AI Diagnostic System shutting down.")


app = FastAPI(
    title="Medical AI Diagnostic System",
    description=(
        "Explainable AI for Medical Image-Based Diagnosis. "
        "3-agent LangGraph pipeline: Medical Image → Visual Explanation → Report."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
        "http://localhost:3002", "http://127.0.0.1:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(sessions.router)
app.include_router(patients.router)
app.include_router(physicians.router)
app.include_router(technicians.router)

# ── Static file serving (heatmaps, reports) ───────────────────────────────────
if os.path.isdir(settings.HEATMAP_DIR):
    app.mount("/heatmaps", StaticFiles(directory=settings.HEATMAP_DIR), name="heatmaps")


@app.get("/", tags=["Health"])
def health_check():
    """
    Simple health-check endpoint.

    Returns:
        dict: Status message to confirm the API is running.
    """
    return {
        "status": "online",
        "service": "Medical AI Diagnostic System",
        "version": "1.0.0",
        "docs": "/docs",
    }
