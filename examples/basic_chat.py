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

# Initialize the client with your API key
client = MegaNova(api_key=api_key)

print("Client initialized successfully.")

# --- Basic Chat Example ---
print("\n--- Basic Chat ---")
try:
    response = client.chat.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What is the capital of France?"}
        ],
        model="meganova-ai/manta-flash-1.0", 
        temperature=0.7,
        max_tokens=100
    )
    print("Response:")
    print(response.choices[0].message.content)
    print(f"Usage: {response.usage}")
except Exception as e:
    print(f"Error during chat: {e}")
