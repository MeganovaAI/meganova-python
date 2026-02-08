"""Subagent orchestration for multi-agent workflows.

Supports:
- Sequential execution: agents run one after another
- Parallel execution: agents run concurrently (via threads)
- Handoff: one agent delegates to another based on the task
"""

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent, AgentResult


@dataclass
class SubagentConfig:
    """Configuration for a subagent in a team."""

    agent: Agent
    role: str = ""
    condition: Optional[Callable[[str], bool]] = None


@dataclass
class TeamResult:
    """Result from running a team of agents."""

    results: List[AgentResult] = field(default_factory=list)
    final_content: str = ""
    agents_used: List[str] = field(default_factory=list)


class AgentTeam:
    """Orchestrate multiple agents working together.

    Usage:
        researcher = Agent(client, model, system_prompt="Research agent...")
        writer = Agent(client, model, system_prompt="Writing agent...")

        team = AgentTeam()
        team.add(researcher, role="researcher")
        team.add(writer, role="writer")

        # Sequential: researcher -> writer
        result = team.run_sequential("Write an article about AI agents")

        # Parallel: both run simultaneously
        result = team.run_parallel("Analyze this from multiple angles")
    """

    def __init__(self) -> None:
        self._agents: List[SubagentConfig] = []

    def add(
        self,
        agent: Agent,
        role: str = "",
        condition: Optional[Callable[[str], bool]] = None,
    ) -> "AgentTeam":
        """Add an agent to the team."""
        self._agents.append(SubagentConfig(agent=agent, role=role, condition=condition))
        return self

    def run_sequential(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> TeamResult:
        """Run agents sequentially, passing each result as context to the next."""
        results: List[AgentResult] = []
        agents_used: List[str] = []
        current_context = context or ""

        for config in self._agents:
            if config.condition and not config.condition(prompt):
                continue

            full_context = current_context
            if results:
                prev = results[-1]
                full_context += f"\n\nPrevious agent ({agents_used[-1]}) output:\n{prev.content}"

            result = config.agent.run(prompt, context=full_context)
            results.append(result)
            agents_used.append(config.role or config.agent.name)
            current_context = full_context

        final = results[-1].content if results else ""

        return TeamResult(
            results=results,
            final_content=final,
            agents_used=agents_used,
        )

    def run_parallel(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_workers: int = 4,
    ) -> TeamResult:
        """Run agents in parallel using threads."""
        eligible = [
            c for c in self._agents
            if c.condition is None or c.condition(prompt)
        ]

        results: List[AgentResult] = []
        agents_used: List[str] = []

        def _run_one(config: SubagentConfig) -> AgentResult:
            return config.agent.run(prompt, context=context)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_one, c): c for c in eligible
            }
            for future in concurrent.futures.as_completed(futures):
                config = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    agents_used.append(config.role or config.agent.name)
                except Exception:
                    pass

        # Combine all results
        combined = "\n\n---\n\n".join(
            f"**{name}**: {r.content}" for name, r in zip(agents_used, results)
        )

        return TeamResult(
            results=results,
            final_content=combined,
            agents_used=agents_used,
        )

    def run_handoff(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> TeamResult:
        """Run the first matching agent based on conditions."""
        for config in self._agents:
            if config.condition and config.condition(prompt):
                result = config.agent.run(prompt, context=context)
                return TeamResult(
                    results=[result],
                    final_content=result.content,
                    agents_used=[config.role or config.agent.name],
                )

        # Fallback to first agent
        if self._agents:
            config = self._agents[0]
            result = config.agent.run(prompt, context=context)
            return TeamResult(
                results=[result],
                final_content=result.content,
                agents_used=[config.role or config.agent.name],
            )

        return TeamResult()
