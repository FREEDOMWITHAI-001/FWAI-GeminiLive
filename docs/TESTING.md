# Testing Details

## Test Phone Numbers

| Name | Phone | Notes |
|------|-------|-------|
| Animesh Mahato | +919600775259 | Founder / Primary tester |

## Quick Test Commands

### Test Intelligence Module Only
```bash
python -c "
import asyncio
from src.services.intelligence import gather_intelligence

async def test():
    result = await gather_intelligence('Animesh', {'company_name': 'Wipro'})
    print(result)

asyncio.run(test())
"
```

### Make Test Call via API
```bash
curl -X POST http://localhost:3000/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumber": "+919600775259",
    "contactName": "Animesh",
    "context": {
      "customer_name": "Animesh",
      "company_name": "Infosys",
      "role": "Software Engineer"
    }
  }'
```

### Make Test Call Without Company (tests Layer 2 only)
```bash
curl -X POST http://localhost:3000/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumber": "+919600775259",
    "contactName": "Animesh"
  }'
```
