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

# --- List Serverless Text Models ---
print("\n--- Serverless Text Generation Models ---")
try:
    result = client.serverless.list_models(modality="text_generation")
    print(f"Found {result.count} models\n")
    for model in result.models:
        print(f"  {model.model_name}")
        if model.model_alias:
            print(f"    Alias: {model.model_alias}")
        if model.cost_per_1k_input is not None:
            print(f"    Input:  ${model.cost_per_1k_input}/1K tokens")
            print(f"    Output: ${model.cost_per_1k_output}/1K tokens")
        if model.tier2_plus_require:
            print(f"    Requires: Tier 2+")
        print()
except Exception as e:
    print(f"Error: {e}")

# --- List Serverless Image Models ---
print("\n--- Serverless Image Generation Models ---")
try:
    result = client.serverless.list_models(modality="text_to_image")
    print(f"Found {result.count} models\n")
    for model in result.models:
        print(f"  {model.model_name}")
        if model.model_alias:
            print(f"    Alias: {model.model_alias}")
        print()
except Exception as e:
    print(f"Error: {e}")
