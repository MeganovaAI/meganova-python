"""Tests for Chat resource."""

import json
from unittest.mock import MagicMock

import pytest

from meganova.models.chat import ChatResponse, ChatStreamChunk
from meganova.resources.chat import Chat, Completions
from tests.conftest import make_chat_response, make_stream_chunk


class TestCompletionsCreate:
    def test_basic_completion(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_chat_response(content="Hi there")
        comp = Completions(mock_sync_transport)

        result = comp.create(messages=[{"role": "user", "content": "Hello"}], model="gpt-4")
        assert isinstance(result, ChatResponse)
        assert result.choices[0].message.content == "Hi there"

    def test_sends_correct_payload(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_chat_response()
        comp = Completions(mock_sync_transport)

        comp.create(messages=[{"role": "user", "content": "Hi"}], model="gpt-4")
        call_kwargs = mock_sync_transport.request.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "gpt-4"
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]
        assert payload["stream"] is False

    def test_temperature_included_when_set(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_chat_response()
        comp = Completions(mock_sync_transport)

        comp.create(messages=[], model="m", temperature=0.7)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["temperature"] == 0.7

    def test_temperature_excluded_when_none(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_chat_response()
        comp = Completions(mock_sync_transport)

        comp.create(messages=[], model="m")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert "temperature" not in payload

    def test_max_tokens_included(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_chat_response()
        comp = Completions(mock_sync_transport)

        comp.create(messages=[], model="m", max_tokens=100)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["max_tokens"] == 100

    def test_tools_included(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_chat_response()
        comp = Completions(mock_sync_transport)

        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        comp.create(messages=[], model="m", tools=tools)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["tools"] == tools

    def test_tool_choice_included(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_chat_response()
        comp = Completions(mock_sync_transport)

        comp.create(messages=[], model="m", tool_choice="auto")
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["tool_choice"] == "auto"

    def test_extra_kwargs_passed(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_chat_response()
        comp = Completions(mock_sync_transport)

        comp.create(messages=[], model="m", top_p=0.9)
        payload = mock_sync_transport.request.call_args.kwargs["json"]
        assert payload["top_p"] == 0.9

    def test_posts_to_correct_path(self, mock_sync_transport):
        mock_sync_transport.request.return_value = make_chat_response()
        comp = Completions(mock_sync_transport)

        comp.create(messages=[], model="m")
        args = mock_sync_transport.request.call_args
        assert args.args == ("POST", "/chat/completions")


class TestCompletionsStreaming:
    def test_stream_returns_iterator(self, mock_sync_transport):
        sse_lines = [
            b"data: " + json.dumps(make_stream_chunk("Hello")).encode(),
            b"data: " + json.dumps(make_stream_chunk(" world")).encode(),
            b"data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_sync_transport.request.return_value = mock_resp

        comp = Completions(mock_sync_transport)
        chunks = list(comp.create(messages=[], model="m", stream=True))
        assert len(chunks) == 2
        assert all(isinstance(c, ChatStreamChunk) for c in chunks)

    def test_stream_sends_stream_flag(self, mock_sync_transport):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter([b"data: [DONE]"])
        mock_sync_transport.request.return_value = mock_resp

        comp = Completions(mock_sync_transport)
        list(comp.create(messages=[], model="m", stream=True))

        call_kwargs = mock_sync_transport.request.call_args.kwargs
        assert call_kwargs["stream"] is True

    def test_stream_skips_empty_lines(self, mock_sync_transport):
        sse_lines = [
            b"",
            b"data: " + json.dumps(make_stream_chunk("Hi")).encode(),
            b"",
            b"data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_sync_transport.request.return_value = mock_resp

        comp = Completions(mock_sync_transport)
        chunks = list(comp.create(messages=[], model="m", stream=True))
        assert len(chunks) == 1

    def test_stream_skips_invalid_json(self, mock_sync_transport):
        sse_lines = [
            b"data: " + json.dumps(make_stream_chunk("Hi")).encode(),
            b"data: {invalid json}",
            b"data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_sync_transport.request.return_value = mock_resp

        comp = Completions(mock_sync_transport)
        chunks = list(comp.create(messages=[], model="m", stream=True))
        assert len(chunks) == 1


class TestChatInit:
    def test_has_completions(self, mock_sync_transport):
        chat = Chat(mock_sync_transport)
        assert isinstance(chat.completions, Completions)
