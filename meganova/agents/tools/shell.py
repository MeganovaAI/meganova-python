"""Shell execution tool for agents.

Executes shell commands either locally or inside a sandbox container.
"""

import subprocess
from typing import Optional

from .base import ToolDefinition


def _execute_shell(
    command: str,
    timeout: int = 30,
    working_dir: Optional[str] = None,
) -> str:
    """Execute a shell command and return output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\nSTDERR: {result.stderr}" if result.stderr else ""
            output += f"\nExit code: {result.returncode}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


shell_tool = ToolDefinition(
    name="execute_shell",
    description="Execute a shell command and return its output. Use for running scripts, installing packages, system commands, etc.",
    func=_execute_shell,
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for the command",
            },
        },
        "required": ["command"],
    },
)
