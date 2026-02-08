"""Knowledge base search tool for agents.

Provides keyword-based search over lorebook-style knowledge entries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ToolDefinition


@dataclass
class KnowledgeEntry:
    """A single knowledge base entry."""

    title: str
    content: str
    keys: List[str] = field(default_factory=list)
    priority: int = 0
    tags: List[str] = field(default_factory=list)


class KnowledgeBase:
    """In-memory keyword-triggered knowledge base (lorebook pattern)."""

    def __init__(self, entries: Optional[List[KnowledgeEntry]] = None) -> None:
        self._entries = entries or []

    def add(self, entry: KnowledgeEntry) -> None:
        self._entries.append(entry)

    def search(self, query: str, max_results: int = 5) -> str:
        """Search entries by keyword matching. Returns formatted results."""
        query_lower = query.lower()
        matches: List[tuple] = []

        for entry in self._entries:
            score = 0
            for key in entry.keys:
                if key.lower() in query_lower:
                    score += 1
            if entry.title.lower() in query_lower:
                score += 2
            if score > 0:
                matches.append((score + entry.priority, entry))

        matches.sort(key=lambda x: x[0], reverse=True)
        top = matches[:max_results]

        if not top:
            return "No relevant knowledge found."

        results = []
        for _, entry in top:
            results.append(f"## {entry.title}\n{entry.content}")

        return "\n\n---\n\n".join(results)

    def to_tool(self, name: str = "search_knowledge_base") -> ToolDefinition:
        """Convert this knowledge base into an agent tool."""
        kb = self

        def _search(query: str, max_results: int = 5) -> str:
            return kb.search(query, max_results)

        return ToolDefinition(
            name=name,
            description="Search the knowledge base for relevant information. Use this when the user asks a question that might be answered by the knowledge base.",
            func=_search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                    },
                },
                "required": ["query"],
            },
        )
