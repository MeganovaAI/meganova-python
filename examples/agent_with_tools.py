"""Example: Agent with custom tools.

This demonstrates creating an agent with @tool decorated functions
that the LLM can call dynamically during conversation.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from meganova import MegaNova
from meganova.agents import Agent, tool


# Initialize client
client = MegaNova(api_key=os.getenv("MEGANOVA_API_KEY", ""))


# Define custom tools using the @tool decorator
@tool("get_weather", "Get the current weather for a city")
def get_weather(city: str, units: str = "celsius") -> str:
    """Simulated weather lookup."""
    # In production, call a real weather API
    return f"Weather in {city}: 22{units[0].upper()}, partly cloudy"


@tool("calculate", "Perform a mathematical calculation")
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math operations are allowed"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Create agent with tools
agent = Agent(
    client=client,
    model="meganova-ai/manta-flash-1.0",
    system_prompt="You are a helpful assistant with access to weather and calculation tools.",
    tools=[get_weather, calculate],
    max_turns=5,
    temperature=0.7,
)

# Run the agent
result = agent.run("What's the weather in Tokyo? Also, what's 15 * 7 + 23?")
print(f"Response: {result.content}")
print(f"Turns: {result.turns}")
print(f"Tool calls: {result.tool_calls_made}")
print(f"Tokens: {result.total_tokens}")
