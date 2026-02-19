-- PostgreSQL schema for FWAI Voice AI Agent
-- Run once: python -m src.db.scripts.migrate

-- calls table
CREATE TABLE IF NOT EXISTS calls (
    call_uuid TEXT PRIMARY KEY,
    plivo_uuid TEXT,
    phone TEXT,
    contact_name TEXT,
    client_name TEXT,
    status TEXT DEFAULT 'pending',
    started_at TEXT,
    ended_at TEXT,
    duration_seconds DOUBLE PRECISION,
    questions_completed INTEGER DEFAULT 0,
    total_questions INTEGER DEFAULT 0,
    transcript TEXT,
    call_summary TEXT,
    interest_level TEXT,
    collected_responses TEXT,
    objections_raised TEXT,
    webhook_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- contact_memory table
CREATE TABLE IF NOT EXISTS contact_memory (
    phone TEXT PRIMARY KEY,
    name TEXT,
    persona TEXT,
    company TEXT,
    role TEXT,
    objections TEXT DEFAULT '[]',
    interest_areas TEXT DEFAULT '[]',
    key_facts TEXT DEFAULT '[]',
    linguistic_style TEXT DEFAULT '{}',
    call_count INTEGER DEFAULT 0,
    last_call_date TEXT,
    last_call_summary TEXT,
    last_call_outcome TEXT,
    all_call_uuids TEXT DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- social_proof_company
CREATE TABLE IF NOT EXISTS social_proof_company (
    company_name TEXT PRIMARY KEY,
    enrollments_count INTEGER DEFAULT 0,
    last_enrollment_date TEXT,
    notable_outcomes TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- social_proof_city
CREATE TABLE IF NOT EXISTS social_proof_city (
    city_name TEXT PRIMARY KEY,
    enrollments_count INTEGER DEFAULT 0,
    trending INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- social_proof_role
CREATE TABLE IF NOT EXISTS social_proof_role (
    role_name TEXT PRIMARY KEY,
    enrollments_count INTEGER DEFAULT 0,
    avg_salary_increase TEXT,
    success_stories TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_calls_created_at ON calls(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(status);
CREATE INDEX IF NOT EXISTS idx_contact_memory_updated ON contact_memory(updated_at DESC);
