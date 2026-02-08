"""Nova Agent SDK — Build AI agents with any OpenAI-compatible model.

Usage:
    from meganova import MegaNova
    from meganova.agents import Agent, tool

    client = MegaNova(api_key="...")

    @tool("greet", "Greet the user by name")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    agent = Agent(
        client=client,
        model="meganova-ai/manta-flash-1.0",
        system_prompt="You are a helpful assistant.",
        tools=[greet],
    )

    result = agent.run("Say hello to Alice")
    print(result.content)
"""

from .agent import Agent, AgentResult, AgentTurnEvent
from .tools.base import tool, ToolDefinition, ToolRegistry
from .hooks import (
    Hook,
    HookContext,
    HookManager,
    HookResult,
    HookType,
    pre_tool_use,
    post_tool_use,
    pre_model_call,
    post_model_call,
    on_error,
)
from .memory import Memory, MessageMemory, SlidingWindowMemory, TokenBudgetMemory
from .subagents import AgentTeam, TeamResult

__all__ = [
    # Core
    "Agent",
    "AgentResult",
    "AgentTurnEvent",
    # Tools
    "tool",
    "ToolDefinition",
    "ToolRegistry",
    # Hooks
    "Hook",
    "HookContext",
    "HookManager",
    "HookResult",
    "HookType",
    "pre_tool_use",
    "post_tool_use",
    "pre_model_call",
    "post_model_call",
    "on_error",
    # Memory
    "Memory",
    "MessageMemory",
    "SlidingWindowMemory",
    "TokenBudgetMemory",
    # Subagents
    "AgentTeam",
    "TeamResult",
]
