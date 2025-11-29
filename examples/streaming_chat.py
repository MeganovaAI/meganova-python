import os
from dotenv import load_dotenv
from meganova import MegaNova
import sys

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    print("Please set MEGANOVA_API_KEY in your .env file.")
    exit(1)

client = MegaNova(api_key=api_key)

# --- Streaming Chat Example ---
print("\n--- Streaming Chat ---")
try:
    print("Response: ", end="", flush=True)
    for chunk in client.chat.stream(
        messages=[
            {"role": "user", "content": "Write a haiku about space."}
        ],
        model="meganova-ai/manta-flash-1.0",
        temperature=0.8
    ):
        content = chunk.choices[0].delta.get("content", "")
        if content:
            print(content, end="", flush=True)
    print("\nStream complete.")
except Exception as e:
    print(f"\nError during streaming: {e}")
