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

## Usage Examples

### 1. Basic Chat Completion

```python
from meganova import MegaNova

client = MegaNova(api_key="YOUR_API_KEY")

response = client.chat.create(
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
response = client.chat.stream(
    messages=[{"role": "user", "content": "Write a haiku about space."}],
    model="meganova-ai/manta-flash-1.0"
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
    print(f"ID: {model.id}, Name: {model.name}, Context: {model.context_length}")
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
