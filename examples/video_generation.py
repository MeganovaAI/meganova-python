import os
from dotenv import load_dotenv
from meganova import MegaNova

load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    exit(1)

client = MegaNova(api_key=api_key)

print("--- Video Generation ---")
try:
    result = client.videos.generate(
        prompt="A cat playing piano in a jazz club",
        model="google/veo-3.1-fast",
        size="1280x720",
        n_seconds=5,
    )
    print(f"Generation ID: {result.id}")
    print(f"Status: {result.status}")

    # Poll until done
    result = client.videos.generate(
        prompt="A cat playing piano in a jazz club",
        model="google/veo-3.1-fast",
        wait=True,
        poll_interval=10,
    )
    print(f"Final status: {result.status}")
    if result.url:
        print(f"Video URL: {result.url}")
except Exception as e:
    print(f"Error: {e}")
