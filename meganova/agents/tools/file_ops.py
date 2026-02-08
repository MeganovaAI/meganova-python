"""File operation tools for agents.

Provides read, write, edit, and list operations on the local filesystem.
"""

import os
from typing import Optional

from .base import ToolDefinition


def _read_file(path: str, max_lines: int = 500) -> str:
    """Read a file and return its contents."""
    try:
        path = os.path.expanduser(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            return "".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        return "".join(lines)
    except Exception as e:
        return f"Error reading {path}: {e}"


def _write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories if needed."""
    try:
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def _edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace old_text with new_text in a file."""
    try:
        path = os.path.expanduser(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if old_text not in content:
            return f"Error: old_text not found in {path}"
        content = content.replace(old_text, new_text, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Edited {path}: replaced text successfully"
    except Exception as e:
        return f"Error editing {path}: {e}"


def _list_directory(path: str = ".", max_entries: int = 200) -> str:
    """List files and directories at the given path."""
    try:
        path = os.path.expanduser(path)
        entries = sorted(os.listdir(path))
        if len(entries) > max_entries:
            entries = entries[:max_entries]
            entries.append(f"... ({len(os.listdir(path)) - max_entries} more)")
        result = []
        for entry in entries:
            full = os.path.join(path, entry)
            prefix = "d " if os.path.isdir(full) else "f "
            result.append(f"{prefix}{entry}")
        return "\n".join(result)
    except Exception as e:
        return f"Error listing {path}: {e}"


read_file_tool = ToolDefinition(
    name="read_file",
    description="Read the contents of a file.",
    func=_read_file,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to read (default: 500)",
            },
        },
        "required": ["path"],
    },
)

write_file_tool = ToolDefinition(
    name="write_file",
    description="Write content to a file. Creates the file and any parent directories if they don't exist.",
    func=_write_file,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
)

edit_file_tool = ToolDefinition(
    name="edit_file",
    description="Edit a file by replacing old_text with new_text. Only replaces the first occurrence.",
    func=_edit_file,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_text": {
                "type": "string",
                "description": "Text to find and replace",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text",
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
)

list_directory_tool = ToolDefinition(
    name="list_directory",
    description="List files and directories at a path. Prefixes with 'd' for directories and 'f' for files.",
    func=_list_directory,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: current directory)",
            },
        },
        "required": [],
    },
)
