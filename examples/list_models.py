from meganova import MegaNova

client = MegaNova(api_key="YOUR_API_KEY")

# --- List Models Example ---
print("\n--- List Models ---")
try:
    models = client.models.list()
    for model in models:
        print(f"ID: {model.id}, Name: {model.model_name}")
        print(f"  Input Price: {model.input_price}, Output Price: {model.output_price}")
        print("-" * 20)
except Exception as e:
    print(f"Error listing models: {e}")

