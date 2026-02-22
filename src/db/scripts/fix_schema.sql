-- Fix schema issues for FWAI Voice AI Agent
-- Run this to fix existing tables

-- Fix updated_at columns - change from TEXT to TIMESTAMP WITH TIME ZONE
ALTER TABLE contact_memory 
    ALTER COLUMN updated_at TYPE TIMESTAMP WITH TIME ZONE USING NULL,
    ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP,
    ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE USING NULL,
    ALTER COLUMN last_call_date TYPE TIMESTAMP WITH TIME ZONE USING NULL;

ALTER TABLE social_proof_company 
    ALTER COLUMN updated_at TYPE TIMESTAMP WITH TIME ZONE USING NULL,
    ALTER COLUMN last_enrollment_date TYPE TIMESTAMP WITH TIME ZONE USING NULL;

ALTER TABLE social_proof_city 
    ALTER COLUMN updated_at TYPE TIMESTAMP WITH TIME ZONE USING NULL;

ALTER TABLE social_proof_role 
    ALTER COLUMN updated_at TYPE TIMESTAMP WITH TIME ZONE USING NULL;

-- Fix calls table timestamp columns
ALTER TABLE calls 
    ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE USING NULL,
    ALTER COLUMN started_at TYPE TIMESTAMP WITH TIME ZONE USING NULL,
    ALTER COLUMN ended_at TYPE TIMESTAMP WITH TIME ZONE USING NULL;

-- Add transcript_entries column for structured transcript data
ALTER TABLE calls 
    ADD COLUMN IF NOT EXISTS transcript_entries JSONB DEFAULT '[]'::jsonb;

-- Add call_metrics column for latency and performance data
ALTER TABLE calls 
    ADD COLUMN IF NOT EXISTS call_metrics JSONB DEFAULT '{}'::jsonb;

-- Add recording_url column
ALTER TABLE calls 
    ADD COLUMN IF NOT EXISTS recording_url TEXT;

-- Add persona column for detected persona per call
ALTER TABLE calls 
    ADD COLUMN IF NOT EXISTS persona TEXT;

-- Update indexes
DROP INDEX IF EXISTS idx_contact_memory_updated;
CREATE INDEX IF NOT EXISTS idx_contact_memory_updated ON contact_memory(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_phone ON calls(phone);

SELECT 'Schema fixes applied successfully!' as result;
