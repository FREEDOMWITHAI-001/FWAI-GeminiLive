import google.auth
from google.auth.transport.requests import Request

creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
creds.refresh(Request())
print(f"Project: {project}")
print(f"Token: {creds.token[:20]}...")
print("Auth OK!")
