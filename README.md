# Meganova Python SDK

Official Python SDK for the [Meganova AI](https://meganova.ai) platform.

## Installation

```bash
pip install meganova
```

## Usage

```python
from meganova import MegaNova

client = MegaNova(api_key="YOUR_API_KEY")

# Chat
response = client.chat.create(
    messages=[{"role": "user", "content": "Hello!"}],
    model="manta-flash"
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.stream(
    messages=[{"role": "user", "content": "Tell me a story."}],
    model="manta-flash"
):
    print(chunk.delta, end="", flush=True)
```

## Features

- Chat Completions (Streaming & Non-streaming)
- Model Management
- Usage & Billing
- Fully typed with Pydantic models


