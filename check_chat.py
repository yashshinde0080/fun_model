
import httpx
import json

key = "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"

def check_chat():
    print("Checking Chat Completion...")
    try:
        client = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "https://test.com",
            },
            timeout=30
        )
        
        payload = {
            "model": "openai/gpt-oss-120b:free",
            "messages": [{"role": "user", "content": "Hello via API"}]
        }
        
        print(f"POST to {client.base_url}chat/completions")
        resp = client.post("/chat/completions", json=payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            print(f"URL: {resp.url}")
        else:
            print("Success!")
            print(resp.json())
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    check_chat()
