
import httpx

client = httpx.Client(base_url="https://openrouter.ai/api/v1")
req = client.build_request("POST", "/chat/completions")
print(f"Base URL: {client.base_url}")
print(f"Full URL: {req.url}")
