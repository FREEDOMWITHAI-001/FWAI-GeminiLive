@echo off
REM WhatsApp Voice Calling with Gemini Live - Startup Script (Windows)

echo ==============================================
echo WhatsApp Voice Calling with Gemini Live
echo ==============================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check for .env file
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please copy .env.example to .env and configure your settings:
    echo   copy .env.example .env
    echo.
)

REM Start the server
echo.
echo Starting server...
python main.py

pause
