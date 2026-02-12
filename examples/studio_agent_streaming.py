"""OpenAI-compatible streaming with a MegaNova Studio agent.

Usage:
    export STUDIO_AGENT_KEY=agent_xxx...
    python examples/studio_agent_streaming.py
"""

import os

from dotenv import load_dotenv

from meganova.studio import StudioAgent

load_dotenv()

api_key = os.getenv("STUDIO_AGENT_KEY")
if not api_key:
    print("Error: STUDIO_AGENT_KEY not found in environment variables.")
    exit(1)

agent = StudioAgent(api_key=api_key)

# ── Non-streaming completions ──
print("=== Non-streaming ===")
response = agent.completions(
    messages=[
        {"role": "user", "content": "What can you help me with? Keep it brief."}
    ],
)
print(f"Model: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens: {response.usage.total_tokens}")
print()

# ── Streaming completions ──
print("=== Streaming ===")
print("Response: ", end="", flush=True)
for chunk in agent.completions(
    messages=[
        {"role": "user", "content": "Write a haiku about AI assistants."}
    ],
    stream=True,
):
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
