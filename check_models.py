
import httpx
import os

key = "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"

def check():
    print("Checking OpenRouter models...")
    try:
        resp = httpx.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            models = resp.json()['data']
            found = False
            for m in models:
                mid = m['id']
                if '120b' in mid or 'gpt-oss' in mid:
                    print(f"Found candidate: {mid}")
                    if mid == "openai/gpt-oss-120b:free":
                        found = True
            
            if found:
                print("TARGET MODEL FOUND.")
            else:
                print("TARGET MODEL NOT FOUND in list.")
        else:
            print(f"Error: {resp.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    check()
