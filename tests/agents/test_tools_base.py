"""Tests for tool system: @tool decorator, ToolDefinition, ToolRegistry, type mapping."""

import pytest

from meganova.agents.tools.base import (
    ToolDefinition,
    ToolRegistry,
    tool,
    _python_type_to_json_schema,
    _build_parameters_schema,
)


class TestToolDefinition:
    def test_basic_creation(self):
        td = ToolDefinition(name="search", description="Search", func=lambda q: q)
        assert td.name == "search"
        assert td.description == "Search"

    def test_execute(self):
        td = ToolDefinition(name="add", description="Add", func=lambda a, b: a + b)
        assert td.execute(a=2, b=3) == 5

    def test_to_openai_tool(self):
        td = ToolDefinition(
            name="f", description="desc", func=lambda: None,
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        oai = td.to_openai_tool()
        assert oai["type"] == "function"
        assert oai["function"]["name"] == "f"
        assert oai["function"]["description"] == "desc"
        assert oai["function"]["parameters"]["properties"]["x"]["type"] == "string"

    def test_default_parameters(self):
        td = ToolDefinition(name="f", description="d", func=lambda: None)
        assert td.parameters == {}


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        td = ToolDefinition(name="f", description="d", func=lambda: None)
        reg.register(td)
        assert reg.get("f") is td

    def test_get_missing_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_list_tools(self):
        reg = ToolRegistry()
        td1 = ToolDefinition(name="a", description="d", func=lambda: None)
        td2 = ToolDefinition(name="b", description="d", func=lambda: None)
        reg.register(td1)
        reg.register(td2)
        assert len(reg.list_tools()) == 2

    def test_len(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(ToolDefinition(name="f", description="d", func=lambda: None))
        assert len(reg) == 1

    def test_contains(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="f", description="d", func=lambda: None))
        assert "f" in reg
        assert "g" not in reg

    def test_to_openai_tools(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="f", description="d", func=lambda: None,
            parameters={"type": "object", "properties": {}},
        ))
        tools = reg.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    def test_register_overwrites(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="f", description="v1", func=lambda: None))
        reg.register(ToolDefinition(name="f", description="v2", func=lambda: None))
        assert reg.get("f").description == "v2"
        assert len(reg) == 1


class TestToolDecorator:
    def test_basic_decorator(self):
        @tool("search", "Search the web")
        def search(query: str) -> str:
            return f"results for {query}"

        assert isinstance(search, ToolDefinition)
        assert search.name == "search"
        assert search.description == "Search the web"

    def test_decorator_infers_name(self):
        @tool()
        def my_func(x: int) -> int:
            return x * 2

        assert my_func.name == "my_func"

    def test_decorator_infers_description_from_docstring(self):
        @tool()
        def my_func(x: int) -> int:
            """Double the input."""
            return x * 2

        assert my_func.description == "Double the input."

    def test_decorator_fallback_description(self):
        @tool()
        def my_func(x: int) -> int:
            return x

        assert "my_func" in my_func.description

    def test_decorator_builds_parameters(self):
        @tool("add", "Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        params = add.parameters
        assert params["type"] == "object"
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert set(params["required"]) == {"a", "b"}

    def test_optional_params_not_required(self):
        @tool()
        def f(x: str, y: int = 5) -> str:
            return x

        assert "x" in f.parameters.get("required", [])
        assert "y" not in f.parameters.get("required", [])

    def test_tool_is_executable(self):
        @tool("mul", "multiply")
        def multiply(a: int, b: int) -> int:
            return a * b

        assert multiply.execute(a=3, b=4) == 12


class TestTypeToJsonSchema:
    def test_str(self):
        assert _python_type_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert _python_type_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _python_type_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list(self):
        assert _python_type_to_json_schema(list) == {"type": "array"}

    def test_dict(self):
        assert _python_type_to_json_schema(dict) == {"type": "object"}

    def test_unknown_defaults_to_string(self):
        class Custom:
            pass
        result = _python_type_to_json_schema(Custom)
        assert result["type"] == "string"


class TestBuildParametersSchema:
    def test_basic(self):
        def f(x: str, y: int) -> str:
            pass
        schema = _build_parameters_schema(f)
        assert schema["type"] == "object"
        assert schema["properties"]["x"]["type"] == "string"
        assert schema["properties"]["y"]["type"] == "integer"
        assert set(schema["required"]) == {"x", "y"}

    def test_with_default(self):
        def f(x: str, y: int = 10) -> str:
            pass
        schema = _build_parameters_schema(f)
        assert schema["required"] == ["x"]

    def test_no_annotations(self):
        def f(x, y):
            pass
        schema = _build_parameters_schema(f)
        assert schema["properties"]["x"]["type"] == "string"

    def test_skips_self(self):
        def f(self, x: str):
            pass
        schema = _build_parameters_schema(f)
        assert "self" not in schema["properties"]
