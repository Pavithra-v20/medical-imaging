@echo off
echo.
echo ============================================================
echo   Medical AI Diagnostic System ^| Backend (FastAPI)
echo ============================================================
echo.
echo  Python env : .venv (Python 3.11 via uv)
echo  URL        : http://localhost:8000
echo  API Docs   : http://localhost:8000/docs
echo.

REM Always use the venv's uvicorn — avoids system Python conflicts
.venv\Scripts\uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
