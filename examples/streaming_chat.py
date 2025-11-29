from meganova import MegaNova
import sys

client = MegaNova(api_key="YOUR_API_KEY")

# --- Streaming Chat Example ---
print("\n--- Streaming Chat ---")
try:
    print("Response: ", end="", flush=True)
    for chunk in client.chat.stream(
        messages=[
            {"role": "user", "content": "Write a haiku about space."}
        ],
        model="manta-flash",
        temperature=0.8
    ):
        content = chunk.choices[0].delta.get("content", "")
        if content:
            print(content, end="", flush=True)
    print("\nStream complete.")
except Exception as e:
    print(f"\nError during streaming: {e}")

