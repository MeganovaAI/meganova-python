import os
from dotenv import load_dotenv
from meganova import MegaNova

load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    exit(1)

client = MegaNova(api_key=api_key)

print("--- Embeddings ---")
try:
    response = client.embeddings.create(
        input="The quick brown fox jumps over the lazy dog",
        model="text-embedding-3-small",
    )
    print(f"Model: {response.model}")
    print(f"Embeddings: {len(response.data)}")
    print(f"Dimensions: {len(response.data[0].embedding)}")
    if response.usage:
        print(f"Tokens used: {response.usage.prompt_tokens}")
except Exception as e:
    print(f"Error: {e}")
