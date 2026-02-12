"""Tool confirmation flow with a MegaNova Cloud agent.

Some agents have tools (e.g. create Freshdesk ticket, send email) that require
user confirmation before executing. This example shows how to handle that.

Usage:
    export STUDIO_AGENT_KEY=agent_xxx...
    python examples/studio_agent_tools.py
"""

import os

from dotenv import load_dotenv

from meganova.cloud import CloudAgent

load_dotenv()

api_key = os.getenv("STUDIO_AGENT_KEY")
if not api_key:
    print("Error: STUDIO_AGENT_KEY not found in environment variables.")
    exit(1)

agent = CloudAgent(api_key=api_key)
conv = agent.conversation()

# Start a conversation that might trigger a tool call
print("You: I need to create a support ticket for a login issue")
response = conv.chat("I need to create a support ticket for a login issue")
print(f"Agent: {response.response}")
print()

# Check if the agent wants to use a tool
if conv.pending_tool_call:
    tool = conv.pending_tool_call
    print(f"--- Tool Confirmation Required ---")
    print(f"Tool: {tool.tool_name}")
    print(f"Action: {tool.description}")
    print(f"Arguments: {tool.tool_arguments}")
    print()

    # Ask the user (in a real app, you'd show a UI prompt)
    user_choice = input("Approve this action? (y/n): ").strip().lower()

    if user_choice == "y":
        result = conv.confirm()
        print(f"Agent: {result.response}")
    else:
        result = conv.reject()
        print(f"Agent: {result.response}")
else:
    print("(No tool confirmation was needed for this conversation)")

print()
print(f"Conversation ID: {conv.conversation_id}")
print(f"Total messages: {len(conv.messages)}")
