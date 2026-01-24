# FWAI Voice Agent Architecture

## System Overview



## Available Tools

| Tool | Trigger Phrases | Action |
|------|-----------------|--------|
| send_whatsapp | "send me a WhatsApp", "message me on WhatsApp" | Sends WhatsApp via Meta API |
| send_sms | "send me an SMS", "text me" | Sends SMS via Plivo |
| send_email | "email me", "send details to my email" | Sends email via SMTP |
| schedule_callback | "call me later", "schedule a call" | Stores callback request |
| book_demo | "book a demo", "schedule a consultation" | Books demo session |

## Call Flow

1. **Call Initiated** - Plivo makes outbound call
2. **Answer** - Server returns greeting XML with GetInput
3. **Speech Loop**:
   - User speaks → Plivo sends POST /plivo/speech
   - Server sends to Gemini with tool definitions
   - Gemini returns text OR tool call
   - If tool call: execute tool, get result, generate response
   - Server returns XML with response + GetInput for next turn
4. **Hangup** - Conversation memory cleared, transcript saved

## Project Structure


