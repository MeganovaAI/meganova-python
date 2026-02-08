"""Example: Customer Support Agent with knowledge base and escalation.

Demonstrates the Agent Mode pattern — a single agent that dynamically
decides whether to search a knowledge base, create a ticket, or escalate.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from meganova import MegaNova
from meganova.agents import Agent, tool
from meganova.agents.tools.knowledge import KnowledgeBase, KnowledgeEntry
from meganova.agents.hooks import HookContext, HookResult, Hook, HookType

# Initialize client
client = MegaNova(api_key=os.getenv("MEGANOVA_API_KEY", ""))

# --- Knowledge Base ---
kb = KnowledgeBase([
    KnowledgeEntry(
        title="Pricing",
        content="Starter: $49/month. Professional: $149/month. Enterprise: Custom. All plans include 14-day free trial.",
        keys=["pricing", "cost", "price", "plan", "how much"],
        priority=10,
    ),
    KnowledgeEntry(
        title="Refund Policy",
        content="Full refund within 30 days of purchase. Pro-rated refunds after 30 days. Contact support to initiate.",
        keys=["refund", "money back", "cancel", "return"],
        priority=10,
    ),
    KnowledgeEntry(
        title="Account Issues",
        content="Reset password at /forgot-password. Locked accounts unlock after 30 minutes. 2FA can be reset by support.",
        keys=["password", "login", "locked", "account", "2FA", "can't sign in"],
        priority=10,
    ),
])

# Convert KB to a tool the agent can use
kb_tool = kb.to_tool("search_support_kb")


# --- Escalation Tool ---
@tool("create_ticket", "Create a support ticket for issues requiring human follow-up")
def create_ticket(subject: str, description: str, priority: str = "medium") -> str:
    """Simulated ticket creation."""
    print(f"  [TICKET CREATED] Subject: {subject}, Priority: {priority}")
    return f"Support ticket created: '{subject}' (priority: {priority}). A team member will follow up within 24 hours."


# --- Audit Hook ---
def audit_tool_use(context: HookContext) -> HookResult:
    """Log every tool call for audit purposes."""
    print(f"  [AUDIT] Tool: {context.tool_name}, Args: {context.tool_args}")
    return HookResult(allow=True)


audit_hook = Hook(HookType.POST_TOOL_USE, audit_tool_use)

# --- Create Agent ---
agent = Agent(
    client=client,
    model="meganova-ai/manta-flash-1.0",
    system_prompt=(
        "You are Nova, a customer support agent. Your goals:\n"
        "1. Search the knowledge base FIRST for any question\n"
        "2. Answer clearly and helpfully based on KB results\n"
        "3. Create a support ticket for issues needing human follow-up\n"
        "4. Never make up information — if unsure, escalate\n"
        "5. Be warm, professional, and concise"
    ),
    tools=[kb_tool, create_ticket],
    hooks=[audit_hook],
    max_turns=5,
    temperature=0.7,
    name="nova-support",
)

# --- Run ---
print("=== Customer Support Agent ===\n")

# Simple KB question
print("User: How much does the professional plan cost?")
result = agent.run("How much does the professional plan cost?")
print(f"Agent: {result.content}\n")

# Question requiring escalation
print("User: I was charged twice for my subscription last month")
result = agent.run("I was charged twice for my subscription last month")
print(f"Agent: {result.content}\n")

print(f"Total tool calls: {result.tool_calls_made}")
