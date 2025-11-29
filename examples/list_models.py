import os
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

# --- List Models Example ---
print("\n--- List Models ---")
try:
    models = client.models.list()
    for model in models:
        print(f"ID: {model.id}")
        if model.name:
            print(f"Name: {model.name}")
        if model.pricing:
            print(f"  Pricing: {model.pricing}")
        print("-" * 20)
except Exception as e:
    print(f"Error listing models: {e}")
