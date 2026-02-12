"""Basic chat with a deployed MegaNova Cloud agent.

Usage:
    export STUDIO_AGENT_KEY=agent_xxx...
    python examples/studio_agent_chat.py
"""

import os

from dotenv import load_dotenv

from meganova.cloud import CloudAgent

load_dotenv()

api_key = os.getenv("STUDIO_AGENT_KEY")
if not api_key:
    print("Error: STUDIO_AGENT_KEY not found in environment variables.")
    print("Set it in your .env file or export it directly.")
    exit(1)

agent = CloudAgent(api_key=api_key)

# ── Get agent info ──
info = agent.info()
print(f"Agent: {info.name}")
print(f"Available: {info.is_available}")
print(f"Welcome: {info.welcome_message}")
print()

# ── Single message ──
print("You: Hello!")
response = agent.chat("Hello!")
print(f"{info.name}: {response.response}")
print(f"  (tokens: {response.tokens_used}, memories: {response.memories_used})")
print()

# ── Multi-turn conversation ──
conv = agent.conversation()
messages = [
    "I need help with my account",
    "My email is user@example.com",
    "I forgot my password",
]

for msg in messages:
    print(f"You: {msg}")
    r = conv.chat(msg)
    print(f"{info.name}: {r.response}")
    print()

print(f"Conversation ID: {conv.conversation_id}")
print(f"Messages exchanged: {len(conv.messages)}")
