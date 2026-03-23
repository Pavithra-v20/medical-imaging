# CT Disease Agent

End-to-end production agent that takes a DICOM CT scan as input and returns a structured disease prediction, radiology report, and PostgreSQL session record.

## Pipeline

```
DICOM (.dcm) → MONAI anonymize & convert → VISTA-3D segmentation
→ CT-CHAT / MedGemma prediction → Report → PostgreSQL sessions table
```

## Quick start

```bash
# 1. Copy env file and fill in your API keys
cp .env.example .env

# 2. Start Postgres + Triton + agent
docker compose up --build

# 3. Run database migration
alembic upgrade head

# 4. Send a CT scan
curl -X POST http://localhost:8080/predict \
  -F "file=@/path/to/scan.dcm" \
  -F "technician_id=tech-01" \
  -F "patient_id=patient-42" \
  -F "physician_id=dr-jones"
```

## Run tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Environment variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL async connection string |
| `NVIDIA_API_KEY` | NVIDIA NIM API key for VISTA-3D |
| `VISTA3D_NIM_URL` | VISTA-3D NIM endpoint |
| `PREDICTION_MODEL` | `ctchat` or `medgemma` |
| `CONFIDENCE_THRESHOLD` | Below this → rev_req = true (default 0.75) |

## Database schema

```sql
sessions (
  session_id    UUID PRIMARY KEY,
  technician_id VARCHAR(64),
  patient_id    VARCHAR(64),
  input         TEXT,   -- DICOM file path
  output        TEXT,   -- JSON prediction result
  summary       TEXT,   -- short summary
  report        TEXT,   -- full radiology report
  physician_id  VARCHAR(64),
  rev_req       BOOLEAN,
  confidence    FLOAT,
  created_at    TIMESTAMPTZ
)
```
