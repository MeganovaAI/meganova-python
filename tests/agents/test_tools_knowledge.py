"""Tests for KnowledgeBase search tool."""

import pytest

from meganova.agents.tools.knowledge import KnowledgeBase, KnowledgeEntry
from meganova.agents.tools.base import ToolDefinition


class TestKnowledgeEntry:
    def test_basic(self):
        e = KnowledgeEntry(title="Python", content="A programming language")
        assert e.title == "Python"
        assert e.keys == []
        assert e.priority == 0

    def test_with_keys(self):
        e = KnowledgeEntry(title="Python", content="...", keys=["python", "py"])
        assert "py" in e.keys


class TestKnowledgeBase:
    def test_empty_search(self):
        kb = KnowledgeBase()
        result = kb.search("anything")
        assert "No relevant knowledge found" in result

    def test_keyword_match(self):
        kb = KnowledgeBase([
            KnowledgeEntry(title="Python", content="A language", keys=["python"]),
            KnowledgeEntry(title="JavaScript", content="A language", keys=["javascript"]),
        ])
        result = kb.search("tell me about python")
        assert "Python" in result
        assert "JavaScript" not in result

    def test_title_match_scores_higher(self):
        kb = KnowledgeBase([
            KnowledgeEntry(title="python basics", content="basics", keys=["programming"]),
            KnowledgeEntry(title="other", content="other", keys=["python"]),
        ])
        result = kb.search("python basics")
        # Title match gives +2, key match gives +1
        assert result.startswith("## python basics")

    def test_priority_boosts_score(self):
        kb = KnowledgeBase([
            KnowledgeEntry(title="Low", content="low priority", keys=["topic"], priority=0),
            KnowledgeEntry(title="High", content="high priority", keys=["topic"], priority=10),
        ])
        result = kb.search("topic")
        # High priority should come first
        lines = result.split("## ")
        assert "High" in lines[1]

    def test_max_results(self):
        entries = [
            KnowledgeEntry(title=f"Entry {i}", content=f"Content {i}", keys=["common"])
            for i in range(10)
        ]
        kb = KnowledgeBase(entries)
        result = kb.search("common", max_results=3)
        assert result.count("## Entry") == 3

    def test_add_entry(self):
        kb = KnowledgeBase()
        kb.add(KnowledgeEntry(title="New", content="new content", keys=["test"]))
        result = kb.search("test")
        assert "New" in result

    def test_case_insensitive_search(self):
        kb = KnowledgeBase([
            KnowledgeEntry(title="Python", content="lang", keys=["Python"]),
        ])
        result = kb.search("PYTHON")
        assert "Python" in result


class TestKnowledgeBaseToTool:
    def test_returns_tool_definition(self):
        kb = KnowledgeBase()
        td = kb.to_tool()
        assert isinstance(td, ToolDefinition)
        assert td.name == "search_knowledge_base"

    def test_custom_name(self):
        kb = KnowledgeBase()
        td = kb.to_tool(name="search_docs")
        assert td.name == "search_docs"

    def test_tool_executes_search(self):
        kb = KnowledgeBase([
            KnowledgeEntry(title="Test", content="test content", keys=["test"]),
        ])
        td = kb.to_tool()
        result = td.execute(query="test")
        assert "Test" in result

    def test_tool_parameters(self):
        kb = KnowledgeBase()
        td = kb.to_tool()
        assert "query" in td.parameters["properties"]
        assert "query" in td.parameters["required"]
