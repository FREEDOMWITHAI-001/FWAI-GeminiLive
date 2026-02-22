-- Idempotent schema fixes for FWAI Voice AI Agent
-- Safe to run on every startup — only adds missing columns, never destroys data

-- Add missing columns to calls table
ALTER TABLE calls ADD COLUMN IF NOT EXISTS transcript_entries JSONB DEFAULT '[]'::jsonb;
ALTER TABLE calls ADD COLUMN IF NOT EXISTS call_metrics JSONB DEFAULT '{}'::jsonb;
ALTER TABLE calls ADD COLUMN IF NOT EXISTS recording_url TEXT;
ALTER TABLE calls ADD COLUMN IF NOT EXISTS persona TEXT;

-- Indexes (idempotent)
CREATE INDEX IF NOT EXISTS idx_calls_phone ON calls(phone);
