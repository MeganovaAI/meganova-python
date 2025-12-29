import os
from dotenv import load_dotenv
from meganova import MegaNova

# Load environment variables from .env
load_dotenv()

# Initialize the client
# The API key is loaded from the environment variable MEGANOVA_API_KEY
# If you didn't set it in .env, you can pass it explicitly:
# client = MegaNova(api_key="your_api_key")
api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    print("Please set MEGANOVA_API_KEY in your .env file.")
    exit(1)

client = MegaNova(api_key=api_key)

print("Sending chat request...")

try:
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        model="meganova-ai/manta-flash-1.0",
        temperature=0.7,
        max_tokens=100
    )

    print("\nResponse:")
    print(f"Role: {response.choices[0].message.role}")
    print(f"Content: {response.choices[0].message.content}")
    print(f"\nUsage: {response.usage}")

except Exception as e:
    print(f"An error occurred: {e}")
