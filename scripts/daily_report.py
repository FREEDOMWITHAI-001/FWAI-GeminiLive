#!/usr/bin/env python3
"""
FWAI Daily Call Report Script
Sends WhatsApp/SMS message with daily call statistics

Usage:
    python daily_report.py          # Send via both WhatsApp and SMS
    python daily_report.py --sms    # Send via SMS only
    python daily_report.py --whatsapp  # Send via WhatsApp only
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import plivo
from dotenv import load_dotenv

# Load environment variables
load_dotenv(project_root / ".env")

# Configuration
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

# Get Meta token (prefer .meta_token file over .env)
def get_meta_token():
    token_file = project_root / ".meta_token"
    if token_file.exists():
        lines = token_file.read_text().strip().split('\n')
        if lines:
            return lines[0]
    return os.getenv("META_ACCESS_TOKEN")

META_ACCESS_TOKEN = get_meta_token()

# Recipients (format: country code + number, e.g., 919052034075)
REPORT_RECIPIENTS = [
    "919052034075",
]


def get_todays_call_count() -> dict:
    """Fetch today's call statistics from Plivo"""
    try:
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        today = datetime.now().strftime('%Y-%m-%d')

        # Get all calls for today (paginate since limit max is 20)
        all_calls = []
        offset = 0
        while True:
            calls = client.calls.list(
                end_time__gte=f'{today} 00:00:00',
                end_time__lte=f'{today} 23:59:59',
                limit=20,
                offset=offset
            )
            if not calls:
                break
            all_calls.extend(calls)
            if len(calls) < 20:
                break
            offset += 20

        total_calls = len(all_calls)

        # Calculate total duration (handle different API response formats)
        total_duration = 0
        for call in all_calls:
            try:
                duration = getattr(call, 'duration', None) or getattr(call, 'bill_duration', 0)
                total_duration += int(duration or 0)
            except:
                pass
        avg_duration = total_duration // total_calls if total_calls > 0 else 0

        return {
            "total_calls": total_calls,
            "total_duration_mins": total_duration // 60,
            "avg_duration_secs": avg_duration,
            "date": datetime.now().strftime('%d %b %Y')
        }
    except Exception as e:
        print(f"Error fetching Plivo stats: {e}")
        return {
            "total_calls": 0,
            "total_duration_mins": 0,
            "avg_duration_secs": 0,
            "date": datetime.now().strftime('%d %b %Y'),
            "error": str(e)
        }


def send_whatsapp(to_number: str, message: str) -> dict:
    """Send WhatsApp message using Meta API"""
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message}
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def send_sms(to_number: str, message: str) -> dict:
    """Send SMS using Plivo API"""
    try:
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        # Format number with + for Plivo
        dst = f"+{to_number}" if not to_number.startswith("+") else to_number

        response = client.messages.create(
            src=PLIVO_PHONE_NUMBER,
            dst=dst,
            text=message
        )
        return {"message_uuid": response.message_uuid, "status": "sent"}
    except Exception as e:
        return {"error": str(e)}


def format_report(stats: dict, short: bool = False) -> str:
    """Format the daily report message

    Args:
        stats: Call statistics dict
        short: If True, return shorter SMS-friendly version
    """
    if short:
        # SMS version (shorter, no emojis for better compatibility)
        return f"""FWAI AI Calling Report - {stats['date']}
Calls: {stats['total_calls']} | Duration: {stats['total_duration_mins']} mins | Avg: {stats['avg_duration_secs']} secs
- AI Calling Platform"""

    # Full version for WhatsApp
    return f"""📈 FWAI Daily AI Calling Report

Date: {stats['date']}
━━━━━━━━━━━━━━━━━━━━━━━━
📞 Calls Handled: {stats['total_calls']}
⏱️ Total Duration: {stats['total_duration_mins']} mins
📊 Avg Call Length: {stats['avg_duration_secs']} secs
━━━━━━━━━━━━━━━━━━━━━━━━
Status: Active ✅

— AI Calling Platform"""


def send_daily_report(use_sms: bool = True, use_whatsapp: bool = True):
    """Main function to send daily report

    Args:
        use_sms: Send via SMS
        use_whatsapp: Send via WhatsApp
    """
    print(f"[{datetime.now()}] Generating daily report...")

    # Get call statistics
    stats = get_todays_call_count()
    print(f"Stats: {stats}")

    # Format messages
    whatsapp_message = format_report(stats, short=False)
    sms_message = format_report(stats, short=True)

    print(f"Message:\n{whatsapp_message}\n")

    # Send to all recipients
    for number in REPORT_RECIPIENTS:
        if use_whatsapp:
            print(f"Sending WhatsApp to {number}...")
            result = send_whatsapp(number, whatsapp_message)
            print(f"WhatsApp Result: {result}")

        if use_sms:
            print(f"Sending SMS to +{number}...")
            result = send_sms(number, sms_message)
            print(f"SMS Result: {result}")

    print("Daily report sent successfully!")


if __name__ == "__main__":
    # Parse command line args
    use_sms = True
    use_whatsapp = True

    if "--sms" in sys.argv:
        use_whatsapp = False
    elif "--whatsapp" in sys.argv:
        use_sms = False

    send_daily_report(use_sms=use_sms, use_whatsapp=use_whatsapp)
