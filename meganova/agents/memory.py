"""Conversation memory management for agents.

Provides different strategies for managing conversation history:
- MessageMemory: Keep all messages (unlimited)
- SlidingWindowMemory: Keep last N messages
- TokenBudgetMemory: Keep messages within a token budget
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import json


@dataclass
class MemoryEntry:
    """A single memory entry."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_message(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls is not None:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        return msg

    @classmethod
    def from_message(cls, msg: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            role=msg["role"],
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
            tool_call_id=msg.get("tool_call_id"),
            name=msg.get("name"),
        )


class Memory(ABC):
    """Base class for memory implementations."""

    @abstractmethod
    def add(self, message: Dict[str, Any]) -> None:
        """Add a message to memory."""

    @abstractmethod
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dict for persistence."""
        return {
            "type": self.__class__.__name__,
            "messages": self.get_messages(),
        }

    @property
    def size(self) -> int:
        return len(self.get_messages())


class MessageMemory(Memory):
    """Keep all messages (unlimited)."""

    def __init__(self) -> None:
        self._entries: List[MemoryEntry] = []

    def add(self, message: Dict[str, Any]) -> None:
        self._entries.append(MemoryEntry.from_message(message))

    def get_messages(self) -> List[Dict[str, Any]]:
        return [e.to_message() for e in self._entries]

    def clear(self) -> None:
        self._entries.clear()


class SlidingWindowMemory(Memory):
    """Keep last N messages, always preserving system messages."""

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages
        self._entries: List[MemoryEntry] = []

    def add(self, message: Dict[str, Any]) -> None:
        self._entries.append(MemoryEntry.from_message(message))

    def get_messages(self) -> List[Dict[str, Any]]:
        if len(self._entries) <= self.max_messages:
            return [e.to_message() for e in self._entries]

        # Always keep system messages, then take the most recent
        system = [e for e in self._entries if e.role == "system"]
        non_system = [e for e in self._entries if e.role != "system"]
        kept = non_system[-(self.max_messages - len(system)) :]
        return [e.to_message() for e in system + kept]

    def clear(self) -> None:
        self._entries.clear()


class TokenBudgetMemory(Memory):
    """Keep messages within a token budget (estimated ~4 chars per token)."""

    CHARS_PER_TOKEN = 4

    def __init__(self, max_tokens: int = 4096) -> None:
        self.max_tokens = max_tokens
        self._entries: List[MemoryEntry] = []

    def add(self, message: Dict[str, Any]) -> None:
        self._entries.append(MemoryEntry.from_message(message))

    def _estimate_tokens(self, entry: MemoryEntry) -> int:
        text = entry.content or ""
        if entry.tool_calls:
            text += json.dumps(entry.tool_calls)
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def get_messages(self) -> List[Dict[str, Any]]:
        # Always keep system messages
        system = [e for e in self._entries if e.role == "system"]
        non_system = [e for e in self._entries if e.role != "system"]

        system_tokens = sum(self._estimate_tokens(e) for e in system)
        budget = self.max_tokens - system_tokens

        # Take messages from the end until budget exhausted
        kept: List[MemoryEntry] = []
        for entry in reversed(non_system):
            cost = self._estimate_tokens(entry)
            if budget - cost < 0 and kept:
                break
            kept.insert(0, entry)
            budget -= cost

        return [e.to_message() for e in system + kept]

    def clear(self) -> None:
        self._entries.clear()
