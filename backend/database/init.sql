-- Medical AI Diagnostic System — Database Initialisation Script
-- Run this once against your PostgreSQL database before starting the server.
-- The FastAPI app also calls `init_db()` on startup which mirrors these CREATE TABLE statements.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- for gen_random_uuid()

-- ─────────────────────────────────────────────────────────────────────────────
-- technicians
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS technicians (
    technician_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) NOT NULL,
    email           VARCHAR(100) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    role            VARCHAR(20) NOT NULL DEFAULT 'technician',  -- 'technician' | 'admin'
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    last_login      TIMESTAMP
);

-- ─────────────────────────────────────────────────────────────────────────────
-- physicians
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS physicians (
    physician_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) NOT NULL,
    specialization  VARCHAR(100) NOT NULL,
    email           VARCHAR(100) UNIQUE NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- patients
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    patient_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(100) NOT NULL,
    dob         DATE NOT NULL,
    gender      VARCHAR(10) NOT NULL,
    contact     VARCHAR(20),
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- diagnostic_sessions
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS diagnostic_sessions (
    session_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Foreign keys
    technician_id           UUID NOT NULL REFERENCES technicians(technician_id),
    patient_id              UUID NOT NULL REFERENCES patients(patient_id),
    physician_id            UUID NOT NULL REFERENCES physicians(physician_id),

    -- Master Agent
    image_path              TEXT,

    -- Agent 1 — Medical Image Agent
    modality                VARCHAR(20),           -- 'xray' | 'ct' | 'mri'
    vista_output            JSONB,                 -- raw model probability scores
    diagnosis_output        JSONB,                 -- full Gemini structured diagnosis
    confidence_score        DECIMAL(5, 2),         -- top class % (e.g. 94.73)
    clinical_notes          TEXT,                  -- Gemini brief clinical note
    model_used              VARCHAR(50) DEFAULT 'ViT-L16-fe + Xception',

    -- Agent 2 — Visual Explanation Agent
    heatmap_path            TEXT,                  -- local path to Grad-CAM PNG
    explanation             TEXT,                  -- Gemini Vision clinical explanation

    -- Agent 3 — Report Agent
    report_link             TEXT,                  -- Google Drive shareable URL

    -- Physician review (nullable until reviewed)
    physician_review_notes  TEXT,
    physician_reviewed_at   TIMESTAMP,

    -- Lifecycle
    status                  VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at              TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for common lookups
CREATE INDEX IF NOT EXISTS idx_sessions_patient    ON diagnostic_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_technician ON diagnostic_sessions(technician_id);
CREATE INDEX IF NOT EXISTS idx_sessions_physician  ON diagnostic_sessions(physician_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status     ON diagnostic_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created    ON diagnostic_sessions(created_at DESC);
