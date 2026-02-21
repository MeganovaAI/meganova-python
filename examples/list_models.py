import os
from dotenv import load_dotenv
from meganova import MegaNova

load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    exit(1)

client = MegaNova(api_key=api_key)

print("--- List Models ---")
try:
    models = client.models.list()
    print(f"Total models: {len(models)}\n")
    for model in models[:10]:
        print(f"  {model.id}")
        if model.owned_by:
            print(f"    Owner: {model.owned_by}")
        if model.capabilities:
            caps = [k for k, v in model.capabilities.items() if v]
            print(f"    Capabilities: {\", \".join(caps)}")
        if model.pricing:
            print(f"    Pricing: {model.pricing}")

    # Get a specific model
    print("\n--- Get Model ---")
    info = client.models.get(models[0].id)
    print(f"Model: {info.id}")
    print(f"  Name: {info.name}")
    print(f"  Context: {info.context_length}")

except Exception as e:
    print(f"Error: {e}")
