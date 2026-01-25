# n8n Flows for FWAI Voice AI

## Outbound Call Flow

Import `outbound_call.json` into n8n to enable:

### 1. Trigger Call from n8n

**Webhook URL:** `POST https://your-n8n-url/webhook/trigger-call`

**Request Body:**
```json
{
  "phoneNumber": "919052034075",
  "contactName": "Kiran",
  "prompt": "You are Vishnu from Freedom with AI...",
  "context": {
    "customer_name": "Kiran",
    "course_name": "Gold Membership",
    "price": "40,000"
  }
}
```

### 2. Receive Call Ended Notification

When call ends, your server sends data to n8n's webhook:

**Webhook URL:** `POST https://your-n8n-url/webhook/call-ended`

**Payload received:**
```json
{
  "event": "call_ended",
  "call_uuid": "abc123-xyz",
  "caller_phone": "+919052034075",
  "duration_seconds": 125.3,
  "timestamp": "2026-01-25T15:30:00.123456",
  "transcript": "[15:04:05] SYSTEM: Call connected\n..."
}
```

## Setup Instructions

1. Import `outbound_call.json` into n8n
2. Update `YOUR_NGROK_URL` in the "Make Outbound Call" node
3. Activate the workflow
4. Copy the webhook URLs from n8n
5. Use the trigger webhook to initiate calls

## Flow Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Webhook Trigger │────>│ Make Outbound    │────>│ Respond Success │
│ /trigger-call   │     │ Call to Server   │     │ or Error        │
└─────────────────┘     └──────────────────┘     └─────────────────┘

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Call Ended      │────>│ Extract Call     │────>│ Process         │
│ Webhook         │     │ Data             │     │ (Add your logic)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Extending the Flow

After "Process Call Ended" node, you can add:
- **Google Sheets**: Log call data
- **Airtable/CRM**: Update contact record
- **Gmail**: Send follow-up email
- **Slack**: Notify team of completed call
- **Database**: Store transcript for analysis
