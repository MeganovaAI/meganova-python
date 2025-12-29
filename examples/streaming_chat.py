import os
from dotenv import load_dotenv
from meganova import MegaNova

load_dotenv()

api_key = os.getenv("MEGANOVA_API_KEY")
if not api_key:
    print("Error: MEGANOVA_API_KEY not found in environment variables.")
    exit(1)

client = MegaNova(api_key=api_key)

print("Streaming chat response...")

try:
    # Use 'stream=True' to get an iterator of chunks
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Write a short haiku about coding."}
        ],
        model="meganova-ai/manta-flash-1.0",
        stream=True
    )

    print("\nResponse: ", end="", flush=True)
    
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta:
            content = chunk.choices[0].delta.get("content", "")
            if content:
                print(content, end="", flush=True)
    
    print("\n\nStream finished.")

except Exception as e:
    print(f"An error occurred: {e}")
