"""Tests for the core Agent class."""

import json
from unittest.mock import MagicMock, patch

import pytest

from meganova.agents.agent import Agent, AgentResult, AgentTurnEvent
from meganova.agents.hooks import Hook, HookContext, HookResult, HookType
from meganova.agents.memory import MessageMemory, SlidingWindowMemory
from meganova.agents.tools.base import ToolDefinition, tool
from meganova.models.chat import (
    ChatChoice,
    ChatMessage,
    ChatResponse,
    FunctionCall,
    TokenUsage,
    ToolCall,
)
from tests.conftest import make_chat_response, make_tool_call


def _chat_response(content="Hello", tool_calls=None, usage=True, finish_reason="stop"):
    """Build a ChatResponse object (not dict)."""
    message = ChatMessage(role="assistant", content=content)
    if tool_calls:
        message = ChatMessage(
            role="assistant", content=None,
            tool_calls=[
                ToolCall(
                    id=tc["id"], type="function",
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in tool_calls
            ],
        )
    choice = ChatChoice(
        index=0, message=message,
        finish_reason="tool_calls" if tool_calls else finish_reason,
    )
    resp = ChatResponse(
        id="chatcmpl-test", object="chat.completion", created=1700000000,
        model="test-model", choices=[choice],
    )
    if usage:
        resp.usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return resp


class TestAgentInit:
    def test_basic_init(self, mock_client):
        agent = Agent(mock_client, model="gpt-4")
        assert agent.model == "gpt-4"
        assert agent.name == "agent"
        assert agent.max_turns == 10

    def test_custom_params(self, mock_client):
        agent = Agent(
            mock_client, model="m", name="custom",
            max_turns=5, temperature=0.7, max_tokens=100,
        )
        assert agent.name == "custom"
        assert agent.max_turns == 5
        assert agent.temperature == 0.7
        assert agent.max_tokens == 100

    def test_tool_definitions_registered(self, mock_client):
        td = ToolDefinition(name="f", description="d", func=lambda: "ok")
        agent = Agent(mock_client, model="m", tools=[td])
        assert "f" in agent._registry

    def test_callable_tools_wrapped(self, mock_client):
        def my_func():
            """A test function."""
            return "result"

        agent = Agent(mock_client, model="m", tools=[my_func])
        assert "my_func" in agent._registry

    def test_hooks_registered(self, mock_client):
        hook = Hook(HookType.PRE_TOOL_USE, lambda ctx: None)
        agent = Agent(mock_client, model="m", hooks=[hook])
        assert agent._hooks.has_hooks(HookType.PRE_TOOL_USE)

    def test_default_memory(self, mock_client):
        agent = Agent(mock_client, model="m")
        assert isinstance(agent.memory, MessageMemory)

    def test_custom_memory(self, mock_client):
        mem = SlidingWindowMemory(max_messages=10)
        agent = Agent(mock_client, model="m", memory=mem)
        assert agent.memory is mem

    def test_metadata_defaults_to_empty_dict(self, mock_client):
        agent = Agent(mock_client, model="m")
        assert agent.metadata == {}


class TestAgentRun:
    def test_simple_response(self, mock_client):
        mock_client.chat.completions.create.return_value = _chat_response("Hello!")
        agent = Agent(mock_client, model="m")

        result = agent.run("Hi")
        assert isinstance(result, AgentResult)
        assert result.content == "Hello!"
        assert result.turns == 1
        assert result.stop_reason == "complete"

    def test_tracks_tokens(self, mock_client):
        mock_client.chat.completions.create.return_value = _chat_response("Hi")
        agent = Agent(mock_client, model="m")

        result = agent.run("Hello")
        assert result.total_tokens == 30

    def test_tool_call_then_response(self, mock_client):
        tool_calls = [make_tool_call(name="get_weather", arguments='{"city":"NYC"}')]
        tc_response = _chat_response(tool_calls=tool_calls)
        final_response = _chat_response("The weather in NYC is sunny")

        mock_client.chat.completions.create.side_effect = [tc_response, final_response]

        td = ToolDefinition(
            name="get_weather", description="Get weather",
            func=lambda city: f"Sunny in {city}",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        agent = Agent(mock_client, model="m", tools=[td])

        result = agent.run("Weather in NYC?")
        assert result.content == "The weather in NYC is sunny"
        assert result.tool_calls_made == 1
        assert result.turns == 2

    def test_unknown_tool_returns_error(self, mock_client):
        tool_calls = [make_tool_call(name="nonexistent")]
        tc_response = _chat_response(tool_calls=tool_calls)
        final_response = _chat_response("I couldn't find that tool")

        mock_client.chat.completions.create.side_effect = [tc_response, final_response]

        agent = Agent(mock_client, model="m")
        result = agent.run("do something")
        assert result.tool_calls_made == 1

    def test_max_turns_reached(self, mock_client):
        tool_calls = [make_tool_call()]
        tc_response = _chat_response(tool_calls=tool_calls)
        mock_client.chat.completions.create.return_value = tc_response

        td = ToolDefinition(name="get_weather", description="d", func=lambda **kw: "result")
        agent = Agent(mock_client, model="m", tools=[td], max_turns=2)

        result = agent.run("loop forever")
        assert result.stop_reason == "max_turns"
        assert result.turns == 2

    def test_llm_error_returns_error_result(self, mock_client):
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        agent = Agent(mock_client, model="m")

        result = agent.run("test")
        assert result.stop_reason == "error"
        assert "Error:" in result.content

    def test_context_appended_to_system_prompt(self, mock_client):
        mock_client.chat.completions.create.return_value = _chat_response("Hi")
        agent = Agent(mock_client, model="m", system_prompt="Base prompt")

        agent.run("Hello", context="Extra context here")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert "Extra context here" in messages[0]["content"]

    def test_tool_args_parsed_from_json(self, mock_client):
        tool_calls = [make_tool_call(name="f", arguments='{"x": 42}')]
        tc_response = _chat_response(tool_calls=tool_calls)
        final_response = _chat_response("done")

        mock_client.chat.completions.create.side_effect = [tc_response, final_response]

        captured_args = {}
        def my_func(**kwargs):
            captured_args.update(kwargs)
            return "ok"

        td = ToolDefinition(name="f", description="d", func=my_func)
        agent = Agent(mock_client, model="m", tools=[td])

        agent.run("test")
        assert captured_args == {"x": 42}

    def test_invalid_json_args_uses_empty_dict(self, mock_client):
        tool_calls = [make_tool_call(name="f", arguments="not json")]
        tc_response = _chat_response(tool_calls=tool_calls)
        final_response = _chat_response("done")

        mock_client.chat.completions.create.side_effect = [tc_response, final_response]

        td = ToolDefinition(name="f", description="d", func=lambda: "ok")
        agent = Agent(mock_client, model="m", tools=[td])

        result = agent.run("test")
        assert result.content == "done"

    def test_tool_error_caught(self, mock_client):
        tool_calls = [make_tool_call(name="f")]
        tc_response = _chat_response(tool_calls=tool_calls)
        final_response = _chat_response("handled error")

        mock_client.chat.completions.create.side_effect = [tc_response, final_response]

        def bad_func(**kw):
            raise ValueError("tool broke")

        td = ToolDefinition(name="f", description="d", func=bad_func)
        agent = Agent(mock_client, model="m", tools=[td])

        result = agent.run("test")
        assert result.content == "handled error"

    def test_sends_tools_to_llm(self, mock_client):
        mock_client.chat.completions.create.return_value = _chat_response("ok")
        td = ToolDefinition(
            name="f", description="desc", func=lambda: None,
            parameters={"type": "object", "properties": {}},
        )
        agent = Agent(mock_client, model="m", tools=[td])

        agent.run("test")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 1

    def test_no_tools_sends_none(self, mock_client):
        mock_client.chat.completions.create.return_value = _chat_response("ok")
        agent = Agent(mock_client, model="m")

        agent.run("test")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] is None

    def test_temperature_forwarded(self, mock_client):
        mock_client.chat.completions.create.return_value = _chat_response("ok")
        agent = Agent(mock_client, model="m", temperature=0.5)

        agent.run("test")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_messages_include_system_and_user(self, mock_client):
        mock_client.chat.completions.create.return_value = _chat_response("ok")
        agent = Agent(mock_client, model="m", system_prompt="Be helpful")

        agent.run("Hello there")
        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        # The user message is the last one passed to the LLM on the first call
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert user_msgs[-1] == {"role": "user", "content": "Hello there"}


class TestAgentHookIntegration:
    def test_pre_tool_hook_blocks_execution(self, mock_client):
        tool_calls = [make_tool_call(name="f")]
        tc_response = _chat_response(tool_calls=tool_calls)
        final_response = _chat_response("blocked")

        mock_client.chat.completions.create.side_effect = [tc_response, final_response]

        td = ToolDefinition(name="f", description="d", func=lambda **kw: "should not run")
        hook = Hook(
            HookType.PRE_TOOL_USE,
            lambda ctx: HookResult(allow=False, reason="denied"),
        )
        agent = Agent(mock_client, model="m", tools=[td], hooks=[hook])

        result = agent.run("test")
        # Tool was denied, agent continued with denial message
        assert result.tool_calls_made == 1

    def test_pre_tool_hook_modifies_args(self, mock_client):
        tool_calls = [make_tool_call(name="f", arguments='{"x": 1}')]
        tc_response = _chat_response(tool_calls=tool_calls)
        final_response = _chat_response("done")

        mock_client.chat.completions.create.side_effect = [tc_response, final_response]

        captured = {}
        def my_func(**kwargs):
            captured.update(kwargs)
            return "ok"

        td = ToolDefinition(name="f", description="d", func=my_func)
        hook = Hook(
            HookType.PRE_TOOL_USE,
            lambda ctx: HookResult(modified_args={"x": 999}),
        )
        agent = Agent(mock_client, model="m", tools=[td], hooks=[hook])

        agent.run("test")
        assert captured["x"] == 999


class TestAgentStreaming:
    def test_streaming_yields_events(self, mock_client):
        mock_client.chat.completions.create.return_value = _chat_response("Hello stream")
        agent = Agent(mock_client, model="m")

        events = list(agent.run("test", stream=True))
        types = [e.type for e in events]
        assert "text" in types
        assert "done" in types

    def test_streaming_tool_calls(self, mock_client):
        tool_calls = [make_tool_call(name="f")]
        tc_response = _chat_response(tool_calls=tool_calls)
        final_response = _chat_response("done")

        mock_client.chat.completions.create.side_effect = [tc_response, final_response]

        td = ToolDefinition(name="f", description="d", func=lambda **kw: "result")
        agent = Agent(mock_client, model="m", tools=[td])

        events = list(agent.run("test", stream=True))
        types = [e.type for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "done" in types

    def test_streaming_error(self, mock_client):
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        agent = Agent(mock_client, model="m")

        events = list(agent.run("test", stream=True))
        assert events[-1].type == "error"

    def test_streaming_max_turns(self, mock_client):
        tool_calls = [make_tool_call(name="f")]
        tc_response = _chat_response(tool_calls=tool_calls)
        mock_client.chat.completions.create.return_value = tc_response

        td = ToolDefinition(name="f", description="d", func=lambda **kw: "result")
        agent = Agent(mock_client, model="m", tools=[td], max_turns=1)

        events = list(agent.run("test", stream=True))
        assert events[-1].type == "done"
        assert "Max turns" in events[-1].content


class TestAgentResult:
    def test_defaults(self):
        result = AgentResult(content="Hi", turns=1)
        assert result.total_tokens == 0
        assert result.tool_calls_made == 0
        assert result.messages == []
        assert result.model == ""
        assert result.stop_reason == "complete"


class TestAgentTurnEvent:
    def test_defaults(self):
        event = AgentTurnEvent(type="text")
        assert event.content == ""
        assert event.tool_name is None
        assert event.turn == 0
