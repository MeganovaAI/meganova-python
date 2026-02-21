import os
import base64
from dotenv import load_dotenv
from meganova import MegaNova

load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    exit(1)

client = MegaNova(api_key=api_key)

print("--- Image Generation (OpenAI-compatible) ---")
try:
    # Using OpenAI-style size parameter
    response = client.images.generate(
        prompt="A cozy cabin in a snowy forest at night, warm light glowing from the windows",
        model="ByteDance/SeedDream-4-5",
        size="1024x1024",
        n=1,
    )

    print(f"Created: {response.created}")
    print(f"Images: {len(response.data)}")

    for i, image in enumerate(response.data):
        if image.b64_json:
            image_bytes = base64.b64decode(image.b64_json)
            filename = f"generated_{i}.png"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            print(f"  Saved: {filename} ({len(image_bytes):,} bytes)")
        elif image.url:
            print(f"  URL: {image.url}")

except Exception as e:
    print(f"Error: {e}")
