# Manual Testing Checklist

Use this checklist for comprehensive manual testing of the FWAI Voice AI Agent.

## Pre-Test Setup

- [ ] Server URL: `http://140.245.206.162:3000`
- [ ] SSH access to OCI server working
- [ ] Test phone number available
- [ ] Plivo console configured with correct URLs
- [ ] n8n workflow imported and activated

---

## 1. Server Health Checks

### From Browser
- [ ] Open `http://140.245.206.162:3000` - should show JSON response
- [ ] Response contains `"status": "ok"`

### From Terminal
```bash
curl http://140.245.206.162:3000
```
- [ ] Returns JSON with status "ok"

### On OCI Server
```bash
sudo systemctl status fwai-app
```
- [ ] Service is "active (running)"

---

## 2. Outbound Call Test

### Make Test Call
```bash
curl -X POST http://140.245.206.162:3000/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "YOUR_PHONE", "contactName": "Test"}'
```

- [ ] API returns success response with call_uuid
- [ ] Phone rings within 10 seconds
- [ ] Answer call and hear AI greeting
- [ ] AI responds to your speech
- [ ] Conversation flows naturally
- [ ] Call ends properly (hang up or AI ends)

### Check Logs During Call
```bash
sudo journalctl -u fwai-app -f
```

- [ ] "Incoming call" log appears
- [ ] "Connecting to Gemini" log appears
- [ ] "Audio stream started" log appears
- [ ] No error messages

### After Call Ends
```bash
ls -la /opt/fwai/FWAI-GeminiLive/transcripts/
```

- [ ] New transcript file created
- [ ] Transcript contains USER/AGENT labels
- [ ] Transcript content matches conversation

---

## 3. Webhook Tests

### WhatsApp Verification
```bash
curl "http://140.245.206.162:3000/webhook?hub.mode=subscribe&hub.verify_token=PSPK&hub.challenge=test123"
```
- [ ] Returns "test123"

### Plivo Webhook (Check in Plivo Console)
- [ ] Answer URL: `http://140.245.206.162:3000/plivo/answer`
- [ ] Hangup URL: `http://140.245.206.162:3000/plivo/hangup`
- [ ] Both URLs return valid responses when tested

---

## 4. n8n Integration Test

### Trigger Call via n8n
```bash
curl -X POST https://YOUR_N8N_URL/webhook/trigger-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "YOUR_PHONE", "contactName": "n8n Test"}'
```

- [ ] n8n workflow triggers
- [ ] Call is made successfully
- [ ] Call data received back in n8n when call ends

### n8n Workflow Check
- [ ] Webhook trigger node working
- [ ] HTTP request to FWAI succeeds
- [ ] Call ended webhook receives data
- [ ] Post-call automation runs (if configured)

---

## 5. AI Agent Quality

During a test call, verify:

- [ ] AI introduces itself correctly (name, company)
- [ ] AI speaks with natural Indian English accent
- [ ] AI waits for user to finish speaking
- [ ] AI handles interruptions gracefully
- [ ] AI asks relevant questions
- [ ] AI handles objections professionally
- [ ] Call ends gracefully when conversation concludes

---

## 6. Error Handling

### Invalid Phone Number
```bash
curl -X POST http://140.245.206.162:3000/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "invalid", "contactName": "Test"}'
```
- [ ] Returns error message (not crash)

### Server Restart Recovery
```bash
sudo systemctl restart fwai-app
```
- [ ] Service restarts without errors
- [ ] Health check works after restart
- [ ] Can make calls after restart

---

## 7. Performance Checks

### Memory Usage
```bash
free -h
```
- [ ] Available memory > 200MB
- [ ] Swap usage reasonable

### Disk Space
```bash
df -h
```
- [ ] Root filesystem not full

### Response Time
```bash
time curl http://140.245.206.162:3000
```
- [ ] Response time < 1 second

---

## 8. Security Checks

- [ ] `.env` file not accessible via HTTP
- [ ] SSH key has correct permissions (400)
- [ ] No sensitive data in logs
- [ ] Webhook verification token working

---

## Test Results

| Test | Pass/Fail | Notes |
|------|-----------|-------|
| Health Check | | |
| Outbound Call | | |
| Webhooks | | |
| n8n Integration | | |
| AI Quality | | |
| Error Handling | | |
| Performance | | |
| Security | | |

**Tested By:** _______________
**Date:** _______________
**Server IP:** 140.245.206.162
