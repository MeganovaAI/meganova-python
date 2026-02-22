"""Tests for memory implementations."""

import pytest

from meganova.agents.memory import (
    Memory,
    MemoryEntry,
    MessageMemory,
    SlidingWindowMemory,
    TokenBudgetMemory,
)


class TestMemoryEntry:
    def test_to_message_basic(self):
        entry = MemoryEntry(role="user", content="Hello")
        msg = entry.to_message()
        assert msg == {"role": "user", "content": "Hello"}

    def test_to_message_with_tool_calls(self):
        entry = MemoryEntry(role="assistant", tool_calls=[{"id": "c1"}])
        msg = entry.to_message()
        assert msg["tool_calls"] == [{"id": "c1"}]

    def test_to_message_with_tool_call_id(self):
        entry = MemoryEntry(role="tool", content="result", tool_call_id="c1")
        msg = entry.to_message()
        assert msg["tool_call_id"] == "c1"

    def test_from_message(self):
        msg = {"role": "user", "content": "Hi", "name": "Alice"}
        entry = MemoryEntry.from_message(msg)
        assert entry.role == "user"
        assert entry.content == "Hi"
        assert entry.name == "Alice"

    def test_roundtrip(self):
        original = {"role": "assistant", "content": "Hello"}
        entry = MemoryEntry.from_message(original)
        result = entry.to_message()
        assert result == original


class TestMessageMemory:
    def test_add_and_get(self):
        mem = MessageMemory()
        mem.add({"role": "user", "content": "Hi"})
        mem.add({"role": "assistant", "content": "Hello"})
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_clear(self):
        mem = MessageMemory()
        mem.add({"role": "user", "content": "Hi"})
        mem.clear()
        assert mem.get_messages() == []

    def test_size(self):
        mem = MessageMemory()
        assert mem.size == 0
        mem.add({"role": "user", "content": "Hi"})
        assert mem.size == 1

    def test_to_dict(self):
        mem = MessageMemory()
        mem.add({"role": "user", "content": "Hi"})
        d = mem.to_dict()
        assert d["type"] == "MessageMemory"
        assert len(d["messages"]) == 1

    def test_unlimited_storage(self):
        mem = MessageMemory()
        for i in range(100):
            mem.add({"role": "user", "content": f"msg {i}"})
        assert mem.size == 100


class TestSlidingWindowMemory:
    def test_under_limit(self):
        mem = SlidingWindowMemory(max_messages=5)
        for i in range(3):
            mem.add({"role": "user", "content": f"msg {i}"})
        assert len(mem.get_messages()) == 3

    def test_over_limit_trims(self):
        mem = SlidingWindowMemory(max_messages=3)
        for i in range(5):
            mem.add({"role": "user", "content": f"msg {i}"})
        msgs = mem.get_messages()
        assert len(msgs) == 3
        # Should keep the most recent
        assert msgs[-1]["content"] == "msg 4"

    def test_preserves_system_messages(self):
        mem = SlidingWindowMemory(max_messages=3)
        mem.add({"role": "system", "content": "You are helpful"})
        for i in range(5):
            mem.add({"role": "user", "content": f"msg {i}"})
        msgs = mem.get_messages()
        # System message + 2 most recent user messages = 3
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 3

    def test_clear(self):
        mem = SlidingWindowMemory(max_messages=5)
        mem.add({"role": "user", "content": "Hi"})
        mem.clear()
        assert mem.get_messages() == []

    def test_to_dict(self):
        mem = SlidingWindowMemory(max_messages=5)
        d = mem.to_dict()
        assert d["type"] == "SlidingWindowMemory"


class TestTokenBudgetMemory:
    def test_under_budget(self):
        mem = TokenBudgetMemory(max_tokens=1000)
        mem.add({"role": "user", "content": "Hello"})
        msgs = mem.get_messages()
        assert len(msgs) == 1

    def test_over_budget_trims(self):
        mem = TokenBudgetMemory(max_tokens=10)  # ~40 chars
        # Each message ~10 chars / 4 = ~2.5 tokens
        for i in range(20):
            mem.add({"role": "user", "content": f"Message number {i} with some text"})
        msgs = mem.get_messages()
        assert len(msgs) < 20

    def test_preserves_system_messages(self):
        mem = TokenBudgetMemory(max_tokens=20)
        mem.add({"role": "system", "content": "System"})
        for i in range(10):
            mem.add({"role": "user", "content": f"Long message text number {i}"})
        msgs = mem.get_messages()
        assert msgs[0]["role"] == "system"

    def test_clear(self):
        mem = TokenBudgetMemory(max_tokens=100)
        mem.add({"role": "user", "content": "Hi"})
        mem.clear()
        assert mem.get_messages() == []

    def test_chars_per_token_constant(self):
        assert TokenBudgetMemory.CHARS_PER_TOKEN == 4

    def test_minimum_one_token(self):
        mem = TokenBudgetMemory(max_tokens=1000)
        mem.add({"role": "user", "content": ""})
        msgs = mem.get_messages()
        assert len(msgs) == 1

    def test_to_dict(self):
        mem = TokenBudgetMemory(max_tokens=100)
        d = mem.to_dict()
        assert d["type"] == "TokenBudgetMemory"
