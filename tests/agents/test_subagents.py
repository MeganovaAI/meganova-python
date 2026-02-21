"""Tests for subagent orchestration."""

from unittest.mock import MagicMock, patch

import pytest

from meganova.agents.agent import Agent, AgentResult
from meganova.agents.subagents import AgentTeam, SubagentConfig, TeamResult


def _make_agent(name="agent", content="result"):
    """Create a mock agent that returns a fixed result."""
    agent = MagicMock(spec=Agent)
    agent.name = name
    agent.run.return_value = AgentResult(content=content, turns=1, model="m")
    return agent


class TestTeamResult:
    def test_defaults(self):
        tr = TeamResult()
        assert tr.results == []
        assert tr.final_content == ""
        assert tr.agents_used == []


class TestAgentTeam:
    def test_add_returns_self(self):
        team = AgentTeam()
        result = team.add(_make_agent())
        assert result is team

    def test_add_multiple(self):
        team = AgentTeam()
        team.add(_make_agent("a")).add(_make_agent("b"))
        assert len(team._agents) == 2


class TestSequentialExecution:
    def test_basic_sequential(self):
        a1 = _make_agent("researcher", "research findings")
        a2 = _make_agent("writer", "final article")

        team = AgentTeam()
        team.add(a1, role="researcher").add(a2, role="writer")

        result = team.run_sequential("Write about AI")
        assert result.final_content == "final article"
        assert len(result.results) == 2
        assert result.agents_used == ["researcher", "writer"]

    def test_sequential_passes_context(self):
        a1 = _make_agent("first", "step 1 result")
        a2 = _make_agent("second", "step 2 result")

        team = AgentTeam()
        team.add(a1, role="first").add(a2, role="second")

        team.run_sequential("task")
        # Second agent should receive first agent's output as context
        second_call = a2.run.call_args
        # Check positional or keyword args for context containing previous result
        context_value = ""
        if second_call.kwargs.get("context"):
            context_value = second_call.kwargs["context"]
        elif len(second_call.args) > 1 and second_call.args[1]:
            context_value = second_call.args[1]
        assert "step 1 result" in context_value

    def test_sequential_with_condition(self):
        a1 = _make_agent("a1", "result 1")
        a2 = _make_agent("a2", "result 2")

        team = AgentTeam()
        team.add(a1, role="always")
        team.add(a2, role="conditional", condition=lambda p: "special" in p)

        result = team.run_sequential("normal task")
        assert len(result.results) == 1
        assert result.agents_used == ["always"]

    def test_sequential_condition_matches(self):
        a1 = _make_agent("a1", "result 1")
        a2 = _make_agent("a2", "result 2")

        team = AgentTeam()
        team.add(a1, role="always")
        team.add(a2, role="conditional", condition=lambda p: "special" in p)

        result = team.run_sequential("special task")
        assert len(result.results) == 2

    def test_sequential_with_initial_context(self):
        a1 = _make_agent("a1", "result")

        team = AgentTeam()
        team.add(a1, role="first")

        team.run_sequential("task", context="initial context")
        call_kwargs = a1.run.call_args
        context_arg = call_kwargs.kwargs.get("context", call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")
        assert "initial context" in context_arg

    def test_uses_agent_name_as_fallback(self):
        a1 = _make_agent("my_agent", "result")

        team = AgentTeam()
        team.add(a1)  # No role specified

        result = team.run_sequential("task")
        assert result.agents_used == ["my_agent"]


class TestParallelExecution:
    def test_basic_parallel(self):
        a1 = _make_agent("a1", "result 1")
        a2 = _make_agent("a2", "result 2")

        team = AgentTeam()
        team.add(a1, role="analyst1").add(a2, role="analyst2")

        result = team.run_parallel("analyze this")
        assert len(result.results) == 2
        assert len(result.agents_used) == 2

    def test_parallel_combines_results(self):
        a1 = _make_agent("a1", "perspective 1")
        a2 = _make_agent("a2", "perspective 2")

        team = AgentTeam()
        team.add(a1, role="a").add(a2, role="b")

        result = team.run_parallel("task")
        assert result.final_content  # Should be non-empty combined content

    def test_parallel_respects_conditions(self):
        a1 = _make_agent("a1", "result 1")
        a2 = _make_agent("a2", "result 2")

        team = AgentTeam()
        team.add(a1, role="always")
        team.add(a2, role="never", condition=lambda p: False)

        result = team.run_parallel("task")
        assert len(result.results) == 1

    def test_parallel_handles_agent_error(self):
        a1 = _make_agent("a1", "result 1")
        a2 = MagicMock(spec=Agent)
        a2.name = "a2"
        a2.run.side_effect = RuntimeError("failed")

        team = AgentTeam()
        team.add(a1, role="ok").add(a2, role="fail")

        result = team.run_parallel("task")
        # Should still get result from a1
        assert len(result.results) == 1


class TestHandoffExecution:
    def test_handoff_to_matching_condition(self):
        a1 = _make_agent("coding", "code result")
        a2 = _make_agent("writing", "text result")

        team = AgentTeam()
        team.add(a1, role="coder", condition=lambda p: "code" in p)
        team.add(a2, role="writer", condition=lambda p: "write" in p)

        result = team.run_handoff("write an essay")
        assert result.final_content == "text result"
        assert result.agents_used == ["writer"]

    def test_handoff_fallback_to_first(self):
        a1 = _make_agent("default", "default result")
        a2 = _make_agent("special", "special result")

        team = AgentTeam()
        team.add(a1, role="default")
        team.add(a2, role="special", condition=lambda p: "xyz" in p)

        result = team.run_handoff("normal task")
        assert result.final_content == "default result"

    def test_handoff_empty_team(self):
        team = AgentTeam()
        result = team.run_handoff("task")
        assert result.final_content == ""
        assert result.results == []
