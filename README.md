# Meganova Python SDK

Official Python SDK for the [Meganova AI](https://meganova.ai) platform, designed for simplicity, robustness, and performance.

## Installation

```bash
pip install -e .
```

*Note: This package requires Python 3.8+*

## Configuration

The SDK defaults to the production API URL: `https://api.meganova.ai/v1`.

You can configure the client using your API key:

```python
from meganova import MegaNova

client = MegaNova(api_key="YOUR_API_KEY")
```

Alternatively, set the `MEGANOVA_API_KEY` environment variable.

## Supported Models

Meganova supports a wide range of models across different modalities:

*   **Text Generation**:
    *   `meganova-ai/manta-mini-1.0` (Fast, 8k context)
    *   `meganova-ai/manta-flash-1.0` (Balanced, 32k context)
    *   `meganova-ai/manta-pro-1.0` (Powerful, 65k context)
    *   `Qwen/Qwen3-235B-A22B-Instruct-2507` (262k context)
    *   `moonshotai/Kimi-K2-Thinking` (Reasoning focused)
*   **Video Generation**:
    *   `Byteplus/seedance-1-0-pro-250528` (Text-to-Video)
*   **Vision**:
    *   `Qwen/Qwen2.5-VL-7B-Instruct`
*   **Audio**:
    *   `Systran/faster-whisper-large-v3` (Speech-to-Text)

Use `client.models.list()` to see the full list of available models and their capabilities.

## Usage Examples

### 1. Basic Chat Completion

```python
from meganova import MegaNova

client = MegaNova(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    model="meganova-ai/manta-flash-1.0",
    temperature=0.7
)

print(response.choices[0].message.content)
print(f"Usage: {response.usage}")
```

### 2. Streaming Chat

```python
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a haiku about space."}],
    model="meganova-ai/manta-flash-1.0",
    stream=True
)

print("Response: ", end="", flush=True)
for chunk in response:
    content = chunk.choices[0].delta.get("content", "")
    if content:
        print(content, end="", flush=True)
print()
```

### 3. Listing Models

```python
models = client.models.list()
for model in models:
    print(f"ID: {model.id}")
    print(f"  Name: {model.name}")
    print(f"  Context: {model.context_length}")
    print(f"  Tags: {model.tags}")
```

### 4. Usage & Billing

```python
# Get usage summary
usage = client.usage.summary(start_date="2024-01-01", end_date="2024-01-31")
print(f"Total Requests: {usage.total_requests}")

# Get billing balance
balance = client.billing.get_balance()
print(f"Current Balance: ${balance.amount} {balance.currency}")
```

## Running Examples

The repository includes ready-to-run examples in the `examples/` directory.

1.  Create a `.env` file in the root directory:
    ```env
    MEGANOVA_API_KEY=your_api_key_here
    ```

2.  Run an example:
    ```bash
    python examples/list_models.py
    python examples/basic_chat.py
    python examples/streaming_chat.py
    ```

## Development & Testing

### Context Length Verification

A utility script is provided to verify the *actual* enforced context length of a model versus its claimed metadata.

```bash
python examples/test_context_length.py meganova-ai/manta-flash-1.0
```

## Features

*   **Chat Completions**: Full support for streaming and non-streaming requests.
*   **Model Management**: List and retrieve details for all available models.
*   **Usage & Billing**: Monitor your usage and check balances programmatically.
*   **Type Safety**: Fully typed with Pydantic models for excellent IDE support.
*   **Robustness**: Built-in retry logic (exponential backoff) and error handling.
