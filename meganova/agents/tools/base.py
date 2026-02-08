"""Tool definition system with @tool decorator for the Nova Agent SDK."""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_type_hints


# Python type -> JSON Schema type mapping
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type."""
    origin = getattr(py_type, "__origin__", None)

    if origin is list or origin is List:
        args = getattr(py_type, "__args__", (Any,))
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}

    if origin is dict or origin is Dict:
        return {"type": "object"}

    if origin is Optional:
        args = getattr(py_type, "__args__", ())
        inner = args[0] if args else Any
        return _python_type_to_json_schema(inner)

    # Handle Optional[X] which is Union[X, None]
    if origin is getattr(type(Optional[str]), "__origin__", None):
        args = getattr(py_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])

    schema_type = _TYPE_MAP.get(py_type, "string")
    return {"type": schema_type}


def _build_parameters_schema(func: Callable) -> Dict[str, Any]:
    """Build OpenAI function calling parameters schema from function signature."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        prop: Dict[str, Any] = {}

        # Get type from annotation
        if name in hints:
            prop.update(_python_type_to_json_schema(hints[name]))
        else:
            prop["type"] = "string"

        # Check for default value
        if param.default is inspect.Parameter.empty:
            required.append(name)

        properties[name] = prop

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


@dataclass
class ToolDefinition:
    """A tool that an agent can call."""

    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool function with the given arguments."""
        return self.func(**kwargs)


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool_def: ToolDefinition) -> None:
        self._tools[tool_def.name] = tool_def

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI function calling format."""
        return [t.to_openai_tool() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """Decorator to register a function as an agent tool.

    Usage:
        @tool("search_kb", "Search the knowledge base for relevant information")
        def search_kb(query: str) -> str:
            return kb.search(query)

        @tool()
        def get_weather(city: str, units: str = "celsius") -> str:
            '''Get current weather for a city.'''
            return weather_api.get(city, units)
    """

    def decorator(func: Callable) -> ToolDefinition:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Execute {tool_name}"
        params = _build_parameters_schema(func)

        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_desc.strip(),
            func=func,
            parameters=params,
        )
        return tool_def

    return decorator
