"""Example: Multi-agent team with sequential and parallel execution.

Demonstrates using AgentTeam for orchestrating multiple specialized agents.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from meganova import MegaNova
from meganova.agents import Agent, AgentTeam

# Initialize client
client = MegaNova(api_key=os.getenv("MEGANOVA_API_KEY", ""))

# --- Create specialized agents ---
researcher = Agent(
    client=client,
    model="meganova-ai/manta-flash-1.0",
    system_prompt=(
        "You are a research analyst. When given a topic, provide a concise "
        "summary of key facts, trends, and data points. Be factual and cite "
        "specifics. Keep responses under 200 words."
    ),
    name="researcher",
)

writer = Agent(
    client=client,
    model="meganova-ai/manta-flash-1.0",
    system_prompt=(
        "You are a content writer. Take research findings and transform them "
        "into engaging, readable content. Use clear language, compelling "
        "structure, and actionable insights. Keep responses under 200 words."
    ),
    name="writer",
)

critic = Agent(
    client=client,
    model="meganova-ai/manta-flash-1.0",
    system_prompt=(
        "You are an editor and critic. Review content for accuracy, clarity, "
        "and engagement. Provide specific, constructive feedback. Suggest "
        "one concrete improvement. Keep responses under 100 words."
    ),
    name="critic",
)

# --- Sequential: Research -> Write -> Review ---
print("=== Sequential Team: Research -> Write -> Review ===\n")

team = AgentTeam()
team.add(researcher, role="researcher")
team.add(writer, role="writer")
team.add(critic, role="critic")

result = team.run_sequential(
    "Write a brief analysis of AI agents in enterprise software (2026 trends)"
)

for i, (name, r) in enumerate(zip(result.agents_used, result.results)):
    print(f"--- {name.upper()} ---")
    print(f"{r.content[:300]}...")
    print()

print(f"Agents used: {result.agents_used}")

# --- Parallel: Multiple perspectives ---
print("\n=== Parallel Team: Multiple Perspectives ===\n")

optimist = Agent(
    client=client,
    model="meganova-ai/manta-flash-1.0",
    system_prompt="You focus on opportunities and positive outcomes. Be concise (under 100 words).",
    name="optimist",
)

realist = Agent(
    client=client,
    model="meganova-ai/manta-flash-1.0",
    system_prompt="You focus on practical considerations and risks. Be concise (under 100 words).",
    name="realist",
)

parallel_team = AgentTeam()
parallel_team.add(optimist, role="optimist")
parallel_team.add(realist, role="realist")

result = parallel_team.run_parallel(
    "Should a startup build their own AI agent framework?"
)

print(result.final_content)
