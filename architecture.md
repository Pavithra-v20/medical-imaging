# MedAI Diagnostics System Architecture

## System Overview
MedAI Diagnostics is a full-stack, AI-powered medical radiology platform. It features a Next.js frontend, a FastAPI backend, a PostgreSQL relational database, and an agentic machine learning pipeline choreographed by LangGraph.

The system is designed to ingest human brain MRI scans (JPEG, PNG, DICOM formats), perform deep neural network classification, generate clinical insights via Large Language Models (Gemini 2.0 Flash), and synthesize the results into a final Clinical PDF Report.

## High-Level Architecture
```mermaid
graph TD
    subgraph Frontend [Client-Side (Next.js)]
        UI[User Interface]
        Upload[Upload Form]
        Dash[Diagnostic Dashboard]
    end

    subgraph Backend [Backend API (FastAPI)]
        Router[API Routers]
        DB[PostgreSQL Database]
    end

    subgraph AI_Pipeline [LangGraph Orchestrator]
        Agent1[Agent 1: MRI Inference & Clinical Notes]
        Agent3[Agent 3: Clinical PDF Report Generator]
    end

    subgraph External_Services [External Engines]
        PyTorch[PyTorch ResNet-18 Model]
        Gemini[Google Gemini 2.0 Flash Vision]
    end

    Upload -- "Scan (JPG/PNG/DCM)" --> Router
    Router -- "Session Context" --> AI_Pipeline
    
    Agent1 -- "Image Tensor" --> PyTorch
    PyTorch -- "Prediction & Confidence" --> Agent1
    Agent1 -- "Prediction context" --> Gemini
    Gemini -- "Clinical Diagnosis Text" --> Agent1

    Agent1 -- "Diagnostic State" --> Agent3
    Agent3 -- "PDF Write" --> DB
    
    AI_Pipeline -- "Completed Session" --> Router
    Router -- "JSON Results" --> Dash
```

## Component Breakdown

### 1. Frontend Layer
- **Framework**: Next.js 14 (App Router) with React.
- **Styling**: Tailwind CSS & Lucide React icons.
- **Functionality**: Providers an interactive interface for technicians and physicians to upload scans (`UploadForm.tsx`), monitor diagnostic history (`SessionHistory.tsx`), and review AI outputs (`ResultPanel.tsx`).

### 2. Backend Layer
- **Framework**: FastAPI (Python).
- **Database**: PostgreSQL bridged via SQLAlchemy (ORM).
- **Core Endpoints**: 
  - `POST /sessions/run`: The main entry point that accepts multipart file uploads, intercepts DICOM conversions (via `pydicom`), establishes a database row (`status="pending"`), and fires the LangGraph pipeline synchronously.

### 3. Machine Learning Pipeline (LangGraph)
The analytical pipeline operates strictly on a two-agent architecture governed by a LangGraph `StateGraph`:
- **Agent 1 (Medical Image Analyst):** Loads the proprietary PyTorch model checkpoint. It processes the raw scan, evaluates classification outputs (Glioma, Meningioma, Pituitary, Normal), and passes those probabilistic outputs into a dynamically assembled prompt for the Gemini 2.0 LLM to generate initial clinical notes.
- **Agent 3 (Report Synthesizer):** Triggers upon Agent 1's completion. It accepts the full state dictionary, aggregates the data into a structured `ReportLab` PDF document (saved to disk), and finalizes the `DiagnosticSession` database row to `status="complete"`. *(Note: Agent 2 is bypassed in this state).*

## Data Flow (Workflow)
1. **Ingestion**: A user uploads an MRI scan via the Next.js UI.
2. **Preprocessing**: FastAPI intercepts the upload. If the file is a DICOM (`.dcm`), it normalizes the pixel array and converts it to a standard JPEG buffer. It then creates a `pending` session in PostgreSQL.
3. **Inference (Agent 1)**: The AI orchestrator passes the image buffer through the PyTorch model to secure the Top-1 classification and confidence score. Gemini writes a preliminary clinical interpretation.
4. **Synthesis (Agent 3)**: A formal Clinical Report PDF is generated and logged. The PostgreSQL row is marked `complete`.
5. **Consumption**: The frontend dashboard polls or receives the completed response and transitions the user from the loading screen to the rich, analytic `ResultPanel`.
