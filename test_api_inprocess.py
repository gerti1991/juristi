import os
os.environ.setdefault("RAG_HEADLESS", "1")

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

print("Testing /health ...")
resp = client.get("/health")
print(resp.status_code, resp.json())

print("Testing /search ...")
resp = client.post("/search", json={
    "query": "sa eshte denimi per plagosje me arme te ftohte",
    "top_k": 3
})
print(resp.status_code)
# Print only a short preview to avoid huge output
text = resp.text
print(text[:800] + ("..." if len(text) > 800 else ""))
