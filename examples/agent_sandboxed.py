"""Example: Agent with sandboxed shell execution.

Demonstrates running an agent with Docker sandbox for safe code execution.
Requires: pip install meganova[agents] and Docker running.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from meganova import MegaNova
from meganova.agents import Agent
from meganova.agents.sandbox.docker_provider import DockerSandbox
from meganova.agents.sandbox.base import SandboxConfig
from meganova.agents.tools.base import ToolDefinition

# Initialize client
client = MegaNova(api_key=os.getenv("MEGANOVA_API_KEY", ""))


# --- Set up sandbox ---
config = SandboxConfig(
    image="python:3.11-slim",
    memory_mb=512,
    cpu_count=1,
    timeout_seconds=300,
    network_enabled=False,
    read_only_root=True,
)

sandbox = DockerSandbox(config)


# Create sandbox-aware tools
def execute_python(code: str) -> str:
    """Execute Python code in the sandbox."""
    sandbox.write_file("/workspace/script.py", code)
    result = sandbox.execute("python /workspace/script.py", timeout=30)
    if result.exit_code != 0:
        return f"Error (exit {result.exit_code}):\n{result.stderr}"
    return result.stdout


def execute_shell(command: str) -> str:
    """Execute a shell command in the sandbox."""
    result = sandbox.execute(command, timeout=30)
    output = result.stdout
    if result.stderr:
        output += f"\nSTDERR: {result.stderr}"
    return output


python_tool = ToolDefinition(
    name="execute_python",
    description="Execute Python code in a sandboxed environment. Use for calculations, data processing, etc.",
    func=execute_python,
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
        },
        "required": ["code"],
    },
)

shell_tool = ToolDefinition(
    name="execute_shell",
    description="Execute a shell command in a sandboxed environment.",
    func=execute_shell,
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute",
            },
        },
        "required": ["command"],
    },
)


# --- Create agent with sandbox ---
agent = Agent(
    client=client,
    model="meganova-ai/manta-flash-1.0",
    system_prompt=(
        "You are a data analysis agent. You can execute Python code and shell "
        "commands in a sandboxed environment. Use execute_python for calculations "
        "and data analysis. Be concise in your responses."
    ),
    tools=[python_tool, shell_tool],
    max_turns=5,
    name="data-analyst",
)


# --- Run with sandbox ---
print("=== Sandboxed Agent Example ===\n")
print("Starting Docker sandbox...")

try:
    sandbox.start()
    print(f"Sandbox running: {sandbox.is_running()}\n")

    result = agent.run(
        "Calculate the first 20 Fibonacci numbers and find which ones are prime."
    )
    print(f"Agent: {result.content}")
    print(f"\nTurns: {result.turns}, Tool calls: {result.tool_calls_made}")
finally:
    print("\nStopping sandbox...")
    sandbox.stop()
    print("Done.")
