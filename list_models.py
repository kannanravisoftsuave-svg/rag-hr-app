import requests, os
key = os.environ.get("OPENROUTER_API_KEY", "")
if not key:
    print("Set OPENROUTER_API_KEY first")
else:
    r = requests.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {key}"})
    models = sorted(m["id"] for m in r.json()["data"] if m["id"].endswith(":free"))
    print("\n".join(models))
