#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Transcription Script
Processes recordings older than 30 minutes, generates transcripts, and archives.

Usage:
    python scripts/batch_transcribe.py

Schedule with cron (every hour):
    0 * * * * cd /home/kiran/FWAI_WebRTC_Gemini/FWAI_WebRTC_Gemini && python scripts/batch_transcribe.py >> logs/transcribe.log 2>&1
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Directories
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
ARCHIVE_DIR = PROJECT_ROOT / "recordings_transcribed"

# Create directories if they don't exist
RECORDINGS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(exist_ok=True)

# Settings
MIN_AGE_MINUTES = 30  # Only process files older than this
WHISPER_MODEL = "tiny"  # tiny, base, small, medium, large


def get_file_age_minutes(file_path: Path) -> float:
    """Get file age in minutes"""
    mtime = file_path.stat().st_mtime
    age_seconds = time.time() - mtime
    return age_seconds / 60


def transcribe_with_whisper(audio_file: Path) -> str:
    """Transcribe audio file using Whisper"""
    try:
        import whisper
        print(f"  Loading Whisper model '{WHISPER_MODEL}'...")
        model = whisper.load_model(WHISPER_MODEL)
        print(f"  Transcribing {audio_file.name}...")
        result = model.transcribe(str(audio_file))
        return result["text"].strip()
    except ImportError:
        print("  ERROR: Whisper not installed. Run: pip install openai-whisper")
        return None
    except Exception as e:
        print(f"  ERROR: Transcription failed: {e}")
        return None


def compress_to_mp3(wav_file: Path) -> Path:
    """Compress WAV to MP3 using ffmpeg"""
    mp3_file = wav_file.with_suffix('.mp3')
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', str(wav_file), '-b:a', '32k', str(mp3_file)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return mp3_file
        else:
            print(f"  WARNING: ffmpeg failed: {result.stderr}")
            return wav_file  # Return original if compression fails
    except FileNotFoundError:
        print("  WARNING: ffmpeg not found, keeping WAV format")
        return wav_file


def process_recording(recording_file: Path):
    """Process a single recording: transcribe and archive"""
    call_uuid = recording_file.stem  # filename without extension
    print(f"\nProcessing: {recording_file.name} (UUID: {call_uuid})")

    # Check if already has transcript
    transcript_file = TRANSCRIPTS_DIR / f"{call_uuid}.txt"

    # Transcribe
    transcript = transcribe_with_whisper(recording_file)

    if transcript:
        # Append Whisper transcript to existing transcript file (if any)
        with open(transcript_file, "a") as f:
            f.write(f"\n--- WHISPER TRANSCRIPTION ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---\n")
            f.write(f"{transcript}\n")
        print(f"  Transcript saved: {transcript_file.name}")
    else:
        print(f"  Skipping transcript (Whisper failed)")

    # Compress to MP3 if it's a WAV
    if recording_file.suffix.lower() == '.wav':
        compressed = compress_to_mp3(recording_file)
        if compressed != recording_file:
            recording_file.unlink()  # Delete original WAV
            recording_file = compressed
            print(f"  Compressed to MP3: {compressed.name}")

    # Move to archive
    archive_path = ARCHIVE_DIR / recording_file.name
    shutil.move(str(recording_file), str(archive_path))
    print(f"  Archived: {archive_path.name}")

    return True


def main():
    print(f"=" * 60)
    print(f"Batch Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)
    print(f"Recordings dir: {RECORDINGS_DIR}")
    print(f"Transcripts dir: {TRANSCRIPTS_DIR}")
    print(f"Archive dir: {ARCHIVE_DIR}")
    print(f"Min age: {MIN_AGE_MINUTES} minutes")
    print(f"Whisper model: {WHISPER_MODEL}")

    # Find all recording files
    recordings = list(RECORDINGS_DIR.glob("*.wav")) + list(RECORDINGS_DIR.glob("*.mp3"))

    if not recordings:
        print(f"\nNo recordings found in {RECORDINGS_DIR}")
        return

    print(f"\nFound {len(recordings)} recording(s)")

    # Filter by age
    eligible = []
    for rec in recordings:
        age = get_file_age_minutes(rec)
        if age >= MIN_AGE_MINUTES:
            eligible.append((rec, age))
        else:
            print(f"  Skipping {rec.name} - too recent ({age:.1f} min old)")

    if not eligible:
        print(f"\nNo recordings older than {MIN_AGE_MINUTES} minutes")
        return

    print(f"\n{len(eligible)} recording(s) ready for processing:")

    # Process each eligible recording
    processed = 0
    failed = 0

    for rec, age in eligible:
        try:
            if process_recording(rec):
                processed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n" + "=" * 60)
    print(f"Complete: {processed} processed, {failed} failed")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
