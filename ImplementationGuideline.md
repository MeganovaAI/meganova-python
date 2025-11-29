# Meganova Python SDK — Implementation Guideline

**Package name:** `meganova`  
**Primary class:** `MegaNova`  
**Language:** Python 3.9+  
**Distribution:** `pip install meganova`

This document defines how to design and implement the **official Python SDK** for the **Meganova AI Character Cloud**. It is meant for SDK developers, not end users.

---

## 1. Goals

The Meganova Python SDK should:

1. **Be simple to use**

   ```python
   from meganova import MegaNova

   client = MegaNova(api_key="YOUR_API_KEY")

   resp = client.chat.create(
       session_id="sess_123",
       messages=[{"role": "user", "content": "Hi!"}],
   )
   print(resp.choices[0].message["content"])
```

2. **Map 1:1 to the REST API**, with:

   * Clear, typed request/response models
   * Consistent error handling (`MeganovaError` hierarchy)
   * Support for streaming responses

3. **Be safe and robust**:

   * Timeouts and retries with exponential backoff
   * Thread-safe client design
   * Ready for server-side usage (web backends, workers, bots)

4. **Be forward-compatible**:

   * Easy to add new endpoints (images, audio, tools, etc.)
   * Minimize breaking changes across minor versions

---

## 2. Package Structure

Recommended layout:

```text
meganova/
  __init__.py
  client.py           # MegaNova main client
  config.py           # Config, constants
  transport.py        # HTTP transport (sync + optional async)
  errors.py           # Error classes
  models/
    __init__.py
    chat.py
    models_.py
    usage.py
    billing.py
    common.py
  resources/
    __init__.py
    chat.py          # ChatResource
    models_.py       # ModelsResource
    usage.py         # UsageResource
    billing.py       # BillingResource
  version.py
```

Top-level `__init__.py` should expose the main entry point:

```python
from .client import MegaNova
from .errors import MeganovaError

__all__ = ["MegaNova", "MeganovaError"]
```

---

## 3. Core Public API

### 3.1 Client Initialization

```python
from meganova import MegaNova

client = MegaNova(
    api_key="YOUR_API_KEY",
    base_url="https://inference.meganova.ai/v1",
    timeout=10.0,           # seconds
    max_retries=3,
    region="auto",          # "auto", "us-east", "eu-west", etc.
    user_agent_extra=None,  # optional extra UA fragment
)
```

**Requirements:**

* `api_key` is required; error early if missing.
* `base_url` default should be production API URL.
* `timeout` applied to each HTTP request.
* `max_retries` used on retriable errors (network, 429, 5xx).
* Add headers:

  * `Authorization: Bearer <API_KEY>`
  * `User-Agent: meganova-python/<version>`
  * `X-MN-Region: <region>` when specified
  * `X-MN-SDK: python`

### 3.2 Resources

`MegaNova` provides resource properties:

```python
client.chat        # ChatResource
client.models      # ModelsResource
client.usage       # UsageResource
client.billing     # BillingResource
```

---

## 4. Chat API

### 4.1 Non-streaming chat

**Method signature:**

```python
class ChatResource:
    def create(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> "ChatResponse":
        ...
```

**Usage:**

```python
resp = client.chat.create(
    messages=[
        {"role": "user", "content": "Hey, how are you?"}
    ],
    model="manta-flash",
    temperature=0.9,
    max_tokens=512,
)

print(resp.choices[0].message["content"])
```

**Response model (`models/chat.py`):**

Use dataclasses or `pydantic` (your choice). Example (dataclass):

```python
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]

@dataclass
class ChatMessage:
    role: Role
    content: str

@dataclass
class ChatChoice:
    index: int
    message: ChatMessage
    finish_reason: str

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class ChatResponse:
    id: str
    choices: list[ChatChoice]
    usage: TokenUsage
    model: str | None = None
    request_id: str | None = None
```

SDK should parse JSON into these objects.

### 4.2 Streaming chat

**Method:**

```python
from collections.abc import Iterator

class ChatResource:
    def stream(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator["ChatStreamChunk"]:
        ...
```

**Usage:**

```python
for chunk in client.chat.stream(
    messages=[{"role": "user", "content": "Describe the city tonight."}],
    model="manta-flash",
):
    print(chunk.delta, end="", flush=True)
```

**Streaming chunk model:**

```python
from dataclasses import dataclass

@dataclass
class ChatStreamChunk:
    delta: str        # the new text piece
    index: int = 0
    request_id: str | None = None
    done: bool = False
```

The resource uses SSE or chunked HTTP to yield chunks.

---

## 5. Models List

### 7.1 List models

```python
class ModelsResource:
    def list(self) -> list["ModelInfo"]:
        ...

    def get(self, model_id: str) -> "ModelInfo":
        ...
```

**Usage:**

```python
for m in client.models.list():
    print(m.id, m.name, m.context_length, m.pricing)
```

ModelInfo example (`models/models_.py`):

```python
from dataclasses import dataclass

@dataclass
class ModelPricing:
    prompt_tokens_per_1k: float
    completion_tokens_per_1k: float

@dataclass
class ModelInfo:
    id: str
    name: str
    context_length: int
    modalities: list[str]
    tags: list[str]
    nsfw_support: str
    status: str
    pricing: ModelPricing | None = None
```

---

## 6. Usage & Billing

### 6.1 Usage summary

```python
class UsageResource:
    def summary(
        self,
        *,
        from_: str,
        to: str,
        group_by: str | None = None,
        user_id: str | None = None,
        api_key: str | None = None,
    ) -> "UsageSummary":
        ...
```

**Usage:**

```python
summary = client.usage.summary(
    from_="2025-11-01T00:00:00Z",
    to="2025-11-30T23:59:59Z",
    group_by="user",
)
```

> Note: Use `from_` (with underscore) in Python to avoid `from` keyword.

### 6.2 Balance

```python
class BillingResource:
    def get_balance(self) -> "BillingBalance":
        ...
```

**Usage:**

```python
balance = client.billing.get_balance()
print(balance.balance, balance.currency, balance.status)
```

---

## 7. Transport Layer

Implement a simple HTTP transport to centralize:

* Base URL
* Authentication
* Timeouts
* Retries
* Error handling
* Headers

### 7.1 Sync transport

`transport.py`:

```python
import requests
from typing import Any

from .errors import MeganovaError, APIError, RateLimitError, AuthenticationError

class SyncTransport:
    def __init__(self, base_url: str, api_key: str, timeout: float, max_retries: int, user_agent: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent

    def request(self, method: str, path: str, *, json: Any | None = None, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
        }

        for attempt in range(self.max_retries + 1):
            try:
                r = requests.request(
                    method=method,
                    url=url,
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    stream=False,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise MeganovaError(f"Network error: {exc}") from exc
                continue

            if 200 <= r.status_code < 300:
                data = r.json()
                # optionally attach request_id from headers
                return data

            # Handle errors
            self._handle_error(r)

        raise MeganovaError("Unexpected transport failure")

    def _handle_error(self, response: requests.Response) -> None:
        status = response.status_code
        try:
            payload = response.json()
        except Exception:
            payload = {}

        err = payload.get("error") or {}
        code = err.get("code") or "unknown_error"
        message = err.get("message") or response.text

        if status == 401:
            raise AuthenticationError(message, code=code, status=status)
        if status == 429:
            raise RateLimitError(message, code=code, status=status)
        if 400 <= status < 500:
            raise APIError(message, code=code, status=status)
        if 500 <= status < 600:
            raise APIError(message, code=code, status=status)

        raise MeganovaError(message)
```

---

## 8. Error Handling

`errors.py`:

```python
class MeganovaError(Exception):
    """Base SDK exception."""

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


class APIError(MeganovaError):
    """Generic API error (4xx, 5xx)."""


class AuthenticationError(MeganovaError):
    """Authentication / authorization error."""


class RateLimitError(MeganovaError):
    """Rate limit exceeded."""
```

Usage in user code:

```python
from meganova import MegaNova, MeganovaError
from meganova.errors import RateLimitError

try:
    resp = client.chat.create(...)
except RateLimitError as e:
    print("Rate-limited, retry later:", e)
except MeganovaError as e:
    print("Meganova API error:", e)
```

---

## 9. Versioning & Compatibility

* Follow **Semantic Versioning**:

  * `MAJOR.MINOR.PATCH`
  * `1.0.0` = first stable.
* Breaking changes only in MAJOR versions.
* Add new endpoints as MINOR.
* Bugfixes as PATCH.

`version.py`:

```python
__version__ = "0.1.0"
```

`__init__.py` should export `__version__`.

---

## 10. Basic Usage Examples (for README)

### 10.1 Simple chat

```python
from meganova import MegaNova

client = MegaNova(api_key="YOUR_API_KEY")

resp = client.chat.create(
    messages=[{"role": "user", "content": "Hi Luna, how are you?"}],
    model="manta-flash"
)

print(resp.choices[0].message["content"])
```

### 10.2 Streaming

```python
for chunk in client.chat.stream(
    messages=[{"role": "user", "content": "Tell me about Neon Harbor tonight."}],
    model="manta-flash"
):
    print(chunk.delta, end="", flush=True)
```

### 10.3 Models & usage

```python
models = client.models.list()
for m in models:
    print(m.id, m.modalities, m.context_length)

balance = client.billing.get_balance()
print("Balance:", balance.balance, balance.currency)
```

---

## 11. Testing Guidelines

* Use **pytest**.
* Mock HTTP with `responses` or `httpretty`.
* Test:

  * Success paths
  * Error mapping (401/429/500)
  * Retry logic
  * Timeout behavior
  * Streaming parsing

Example test:

```python
def test_chat_create_success(client, requests_mock):
    requests_mock.post(
        "https://inference.meganova.ai/v1/chat/completions",
        json={
            "id": "msg_123",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
        status_code=200,
    )

    resp = client.chat.create(
        messages=[{"role": "user", "content": "Hi"}],
        model="manta-flash"
    )
    assert resp.choices[0].message.content == "Hello!"
```

---

## 12. Future Extensions

Keep the design ready for:

* `client.characters`
* `client.sessions`
* `client.limits`
* `client.health`
* `client.images.generate(...)`
* `client.audio.tts(...)`
* `client.audio.stt(...)`
* Tools / function calling
* Async client: `AsyncMegaNova`
