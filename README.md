# MedAI Diagnostics System

A full-stack, AI-powered medical radiology platform. It features a Next.js frontend, a FastAPI backend, a PostgreSQL relational database, and an agentic machine learning pipeline choreographed by LangGraph. 

## Features
- **Multi-Format Support**: Directly upload JPEG, PNG, or raw DICOM (`.dcm`) MRI scans.
- **Agentic AI Pipeline**: LangGraph manages a multi-agent orchestration sequence (ResNet ML classification → Gemini 2.0 Clinical Synthesis → PDF Report Generation).
- **Modern Dashboard**: A highly responsive Next.js and Tailwind CSS frontend designed explicitly for radiologists and medical technicians.
- **PostgreSQL Persistence**: Comprehensive diagnostic session history, patient records, and physician metadata tracking.

## Architecture
See the full architectural breakdown and workflow diagram in [architecture.md](./architecture.md).

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL Server

### Backend Setup
1. Clone the repository and navigate to the project root.
2. Initialize the Python virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and populate your database URI and `GEMINI_API_KEY`.
4. Start the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup
1. Navigate to the `frontend/` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Ensure `.env.local` points `NEXT_PUBLIC_API_URL` to `http://localhost:8000`.
4. Start the development server:
   ```bash
   npm run dev
   ```

## Repository State Note
*This documentation reflects the system state at the fundamental `First commit`, executing the 2-Agent pipeline (Model Inference & Report Generation) natively.*