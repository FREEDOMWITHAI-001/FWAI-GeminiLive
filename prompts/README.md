# Client Prompts Directory

This directory contains client-specific AI prompts for different call purposes.

## File Naming
- `{client_name}_prompt.txt` - Main prompt file
- `{client_name}_prompt_questions.txt` - Alternative naming
- `{client_name}.txt` - Short form

## Available Placeholders
Use `{{placeholder}}` in your prompts - they get replaced with context values:

| Placeholder | Description | Default |
|------------|-------------|---------|
| `{{agent_name}}` | AI agent's name | Rahul |
| `{{company_name}}` | Company name | Freedom with AI |
| `{{location}}` | Office location | Hyderabad |
| `{{customer_name}}` | Lead/customer name | there |
| `{{event_name}}` | Event/webinar name | AI Masterclass |
| `{{event_host}}` | Event host | Avinash Mada |
| `{{product_name}}` | Product/service name | AI Upskilling Program |
| `{{product_description}}` | Product details | (see defaults) |
| `{{price}}` | Pricing info | 40,000 rupees |

## Example Use Cases

### 1. Sales/Lead Qualification (fwai_prompt.txt)
- Qualify leads from webinars
- 12 discovery questions
- Objection handling
- Closing

### 2. Onboarding (onboarding_prompt.txt)
- Welcome new customers
- Collect setup info
- Explain features
- Schedule orientation

### 3. Support/Feedback (support_prompt.txt)
- Check satisfaction
- Resolve issues
- Collect feedback

### 4. Appointment Booking (appointment_prompt.txt)
- Confirm availability
- Book appointments
- Send reminders

## API Usage

```json
POST /call/conversational
{
  "phoneNumber": "+919876543210",
  "contactName": "Amit Kumar",
  "clientName": "fwai",
  "n8nWebhookUrl": "https://...",
  "context": {
    "agent_name": "Priya",
    "company_name": "TechCorp",
    "event_name": "Product Demo"
  }
}
```

## Creating New Client Prompts

1. Copy `template_prompt.txt` to `{client}_prompt.txt`
2. Customize the conversation flow
3. Use placeholders for dynamic values
4. Test with the API
