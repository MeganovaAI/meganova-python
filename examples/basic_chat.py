from meganova import MegaNova

# Initialize the client with your API key
client = MegaNova(api_key="YOUR_API_KEY")

print("Client initialized successfully.")

# --- Basic Chat Example ---
print("\n--- Basic Chat ---")
try:
    response = client.chat.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What is the capital of France?"}
        ],
        model="manta-flash", # Replace with a valid model name if needed
        temperature=0.7,
        max_tokens=100
    )
    print("Response:")
    print(response.choices[0].message.content)
    print(f"Usage: {response.usage}")
except Exception as e:
    print(f"Error during chat: {e}")

