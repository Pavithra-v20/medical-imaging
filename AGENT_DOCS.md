# CT Disease Agent Documentation

## Overview
The CT Disease Agent is an AI-driven clinical pipeline for automated radiology segmentation and disease prediction. It takes a DICOM CT scan as input, performs anonymization, converts it to NIfTI format, executes 3D segmentation using NVIDIA's VISTA-3D NIM, generates a disease prediction using LLMs (CT-CHAT or MedGemma), builds a radiology report, and persists the results in a PostgreSQL database.

## Architecture
The agent is built with FastAPI and follows a modular design:

- **Ingestion**: Handles DICOM loading, PHI anonymization, and NIfTI conversion.
- **Segmentation**: Interfaces with NVIDIA VISTA-3D NIM for multi-organ and lesion segmentation.
- **Prediction**: Routes clinical findings to specialized LLMs (CT-CHAT or MedGemma) for disease classification.
- **Reporting**: Generates plain-language summaries and structured radiology reports.
- **Database**: Persists every session with full metadata for review and audit.

## Key Components

### 1. Ingestion (`app/ingestion/`)
- `dicom_loader.py`: Loads DICOM files and verifies modality (CT).
- `anonymizer.py`: Strips PHI tags (Patient Name, ID, Birth Date) to ensure HIPAA compliance.
- `converter.py`: Converts DICOM pixel data to NIfTI format, applying Hounsfield Unit rescaling and building affine matrices from spatial metadata.
- `preprocessor.py`: Normalizes and prepares the NIfTI volume for segmentation.

### 2. Segmentation (`app/segmentation/`)
- `vista3d_client.py`: Async client for the NVIDIA VISTA-3D NIM. It supports "everything" mode to segment 127 structures (organs and lesions).
- `mask_parser.py`: Analyzes the 3D segmentation mask to extract metrics such as lesion count, max lesion size, and organ volumes.

### 3. Prediction (`app/prediction/`)
- `predictor.py`: Orchestrator that chooses between CT-CHAT and MedGemma based on configuration.
- `ct_chat_client.py`: Client for the CT-CHAT model.
- `medgemma_client.py`: Client for the MedGemma model.

### 4. Reporting (`app/reporting/`)
- `report_builder.py`: Constructs a structured radiology report including Patient ID, Session ID, Findings, and Impression.
- `summarizer.py`: Creates a concise, plain-language summary of the AI findings.
- `rev_req_rules.py`: Implements logic to flag sessions for human physician review (e.g., low confidence or critical lesions).

### 5. Database (`app/database/`)
- `models.py`: SQLAlchemy models for the `sessions` table.
- `session_logger.py`: Async logger that writes session records to PostgreSQL.
- `review_queue.py`: Logic for pushing high-risk cases to a physician review queue.

## Data Flow
1. **POST /predict**: Receives a DICOM file and technician/patient/physician IDs.
2. **Ingestion**: DICOM is saved, loaded, anonymized, and converted to NIfTI.
3. **Segmentation**: VISTA-3D NIM returns a 127-class 3D mask.
4. **Metric Extraction**: Mask parser extracts lesion and organ metrics.
5. **Prediction**: LLM receives metrics and generates a disease label and reasoning.
6. **Reporting**: Report and summary are built; review requirement is evaluated.
7. **Persistence**: All data is logged to PostgreSQL.
8. **Response**: Returns session ID, summary, report, and prediction result.

## Configuration
Configuration is managed via environment variables (see `.env.example`):
- `DATABASE_URL`: PostgreSQL connection string.
- `NVIDIA_API_KEY`: API key for NVIDIA NIM services.
- `VISTA3D_NIM_URL`: Endpoint for VISTA-3D segmentation.
- `PREDICTION_MODEL`: Chosen LLM (`ctchat` or `medgemma`).
- `CONFIDENCE_THRESHOLD`: Threshold for flagging physician review.

## Error Handling
The agent uses custom exceptions (`IngestionError`, `SegmentationError`, `PredictionError`) to provide clear feedback on where a pipeline failure occurred.

## Testing
Comprehensive test suite located in `tests/`:
- `test_ingestion.py`: Validates anonymization and conversion.
- `test_segmentation.py`: Tests mask parsing and metric extraction.
- `test_prediction.py`: Verifies model routing.
- `test_reporting.py`: Checks report generation and review rules.
- `test_database.py`: Ensures session logging works correctly.

Run tests using:
```bash
python -m pytest
```
