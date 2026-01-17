#!/bin/bash

# WhatsApp Voice Calling with Gemini Live - Startup Script

echo "=============================================="
echo "WhatsApp Voice Calling with Gemini Live"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found!"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    echo ""
fi

# Start the server
echo ""
echo "Starting server..."
python main.py
