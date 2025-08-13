import requests

BASE = "http://127.0.0.1:8000"

print("/health ->")
try:
    r = requests.get(f"{BASE}/health", timeout=10)
    print(r.status_code, r.text[:200])
except Exception as e:
    print("Health error:", e)

print("/search ->")
try:
    r = requests.post(f"{BASE}/search", json={"query": "sa eshte denimi per plagosje me arme te ftohte", "top_k": 3}, timeout=30)
    print(r.status_code)
    print((r.text or "")[:500])
except Exception as e:
    print("Search error:", e)
