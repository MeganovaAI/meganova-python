import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
base_url = "https://inference.meganova.ai/v1"

print(f"Probing {base_url}/models ...")
try:
    resp = requests.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        print("Response:", resp.json())
    else:
        print("Error:", resp.text)
except Exception as e:
    print(f"Exception: {e}")

