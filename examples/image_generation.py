import os
import base64
from dotenv import load_dotenv
from meganova import MegaNova

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    print("Please set MEGANOVA_API_KEY in your .env file.")
    exit(1)

client = MegaNova(api_key=api_key)

# --- Generate an Image ---
print("\n--- Image Generation ---")
prompt = "A cozy cabin in a snowy forest at night, warm light glowing from the windows, digital art"
print(f"Prompt: {prompt}")
print("Generating image...")

try:
    response = client.images.generate(
        prompt=prompt,
        model="ByteDance/SeedDream-4-5",
        width=1024,
        height=1024,
        num_steps=4,
        guidance_scale=3.5,
    )

    print(f"Received {len(response.data)} image(s)")

    for i, image in enumerate(response.data):
        filename = f"generated_{i}.png"
        image_bytes = base64.b64decode(image.b64_json)
        with open(filename, "wb") as f:
            f.write(image_bytes)
        print(f"  Saved: {filename} ({len(image_bytes):,} bytes)")

except Exception as e:
    print(f"Error: {e}")
