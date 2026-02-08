"""Core Agent class implementing the function-calling agent loop.

The agent loop:
1. Send messages + tool definitions to LLM
2. If LLM returns tool_calls, execute them
3. Feed results back to LLM
4. Repeat until final text response or max_turns
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from ..client import MegaNova
from ..models.chat import ChatResponse
from .hooks import Hook, HookContext, HookManager, HookResult, HookType
from .memory import Memory, MessageMemory
from .tools.base import ToolDefinition, ToolRegistry


@dataclass
class AgentResult:
    """Result from running an agent."""

    content: str
    turns: int
    total_tokens: int = 0
    tool_calls_made: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    stop_reason: str = "complete"


@dataclass
class AgentTurnEvent:
    """Event emitted during agent execution (for streaming)."""

    type: str  # "text", "tool_call", "tool_result", "error", "done"
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    turn: int = 0


class Agent:
    """An AI agent that can use tools to accomplish tasks.

    Usage:
        from meganova import MegaNova
        from meganova.agents import Agent, tool

        client = MegaNova(api_key="...")

        @tool("search", "Search for information")
        def search(query: str) -> str:
            return "search results..."

        agent = Agent(
            client=client,
            model="meganova-ai/manta-flash-1.0",
            system_prompt="You are a helpful assistant.",
            tools=[search],
        )

        result = agent.run("Find information about Python")
        print(result.content)
    """

    def __init__(
        self,
        client: MegaNova,
        model: str,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[Union[ToolDefinition, Callable]]] = None,
        hooks: Optional[List[Hook]] = None,
        memory: Optional[Memory] = None,
        max_turns: int = 10,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        name: str = "agent",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = name
        self.metadata = metadata or {}

        # Set up tool registry
        self._registry = ToolRegistry()
        if tools:
            for t in tools:
                if isinstance(t, ToolDefinition):
                    self._registry.register(t)
                elif callable(t):
                    # Wrap plain callable as a tool
                    tool_def = ToolDefinition(
                        name=t.__name__,
                        description=t.__doc__ or f"Execute {t.__name__}",
                        func=t,
                        parameters={"type": "object", "properties": {}},
                    )
                    self._registry.register(tool_def)

        # Set up hook manager
        self._hooks = HookManager()
        if hooks:
            for hook in hooks:
                self._hooks.register(hook.hook_type, hook.func, hook.priority)

        # Set up memory
        self.memory = memory or MessageMemory()

    def run(
        self,
        prompt: str,
        context: Optional[str] = None,
        stream: bool = False,
    ) -> Union[AgentResult, Iterator[AgentTurnEvent]]:
        """Run the agent with a user prompt.

        Args:
            prompt: The user's message/task.
            context: Optional additional context to prepend.
            stream: If True, yields AgentTurnEvent objects.

        Returns:
            AgentResult with the final response, or iterator of events if streaming.
        """
        if stream:
            return self._run_streaming(prompt, context)
        return self._run_sync(prompt, context)

    def _run_sync(self, prompt: str, context: Optional[str] = None) -> AgentResult:
        """Execute the agent loop synchronously."""
        # Build initial messages
        messages = self._build_initial_messages(prompt, context)

        total_tokens = 0
        tool_calls_made = 0

        for turn in range(self.max_turns):
            # Pre-model-call hook
            hook_ctx = HookContext(
                agent_name=self.name,
                turn=turn,
                messages=messages,
            )
            hook_result = self._hooks.run(HookType.PRE_MODEL_CALL, hook_ctx)
            if hook_result.modified_messages:
                messages = hook_result.modified_messages

            # Call the LLM
            try:
                response = self._call_llm(messages)
            except Exception as e:
                self._hooks.run(
                    HookType.ON_ERROR,
                    HookContext(agent_name=self.name, turn=turn, error=e),
                )
                return AgentResult(
                    content=f"Error: {e}",
                    turns=turn + 1,
                    total_tokens=total_tokens,
                    tool_calls_made=tool_calls_made,
                    messages=messages,
                    model=self.model,
                    stop_reason="error",
                )

            # Post-model-call hook
            self._hooks.run(
                HookType.POST_MODEL_CALL,
                HookContext(
                    agent_name=self.name, turn=turn, response=response
                ),
            )

            # Track token usage
            if response.usage:
                total_tokens += response.usage.total_tokens

            choice = response.choices[0]
            assistant_msg = choice.message

            # Add assistant message to history
            msg_dict = self._message_to_dict(assistant_msg)
            messages.append(msg_dict)

            # Check if LLM wants to call tools
            if choice.finish_reason == "tool_calls" or assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls or []:
                    tool_calls_made += 1

                    # Parse tool arguments
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    # Pre-tool-use hook
                    pre_hook = self._hooks.run(
                        HookType.PRE_TOOL_USE,
                        HookContext(
                            agent_name=self.name,
                            turn=turn,
                            tool_name=tc.function.name,
                            tool_args=args,
                        ),
                    )
                    if not pre_hook.allow:
                        # Hook denied tool execution
                        tool_result = f"Tool execution denied: {pre_hook.reason or 'blocked by hook'}"
                    else:
                        if pre_hook.modified_args:
                            args = pre_hook.modified_args
                        tool_result = self._execute_tool(tc.function.name, args)

                    # Post-tool-use hook
                    self._hooks.run(
                        HookType.POST_TOOL_USE,
                        HookContext(
                            agent_name=self.name,
                            turn=turn,
                            tool_name=tc.function.name,
                            tool_args=args,
                            tool_result=tool_result,
                        ),
                    )

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "content": str(tool_result),
                            "tool_call_id": tc.id,
                        }
                    )

                continue  # LLM gets another turn

            # LLM is done — return final response
            content = assistant_msg.content or ""

            # Store in memory
            for msg in messages:
                self.memory.add(msg)

            self._hooks.run(
                HookType.ON_COMPLETE,
                HookContext(
                    agent_name=self.name,
                    turn=turn,
                    response=content,
                ),
            )

            return AgentResult(
                content=content,
                turns=turn + 1,
                total_tokens=total_tokens,
                tool_calls_made=tool_calls_made,
                messages=messages,
                model=self.model,
                stop_reason="complete",
            )

        # Max turns reached
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_content = msg["content"]
                break

        return AgentResult(
            content=last_content or "Max turns reached without completion.",
            turns=self.max_turns,
            total_tokens=total_tokens,
            tool_calls_made=tool_calls_made,
            messages=messages,
            model=self.model,
            stop_reason="max_turns",
        )

    def _run_streaming(
        self, prompt: str, context: Optional[str] = None
    ) -> Iterator[AgentTurnEvent]:
        """Execute the agent loop with streaming events."""
        messages = self._build_initial_messages(prompt, context)
        total_tokens = 0

        for turn in range(self.max_turns):
            try:
                response = self._call_llm(messages)
            except Exception as e:
                yield AgentTurnEvent(
                    type="error", content=str(e), turn=turn
                )
                return

            if response.usage:
                total_tokens += response.usage.total_tokens

            choice = response.choices[0]
            assistant_msg = choice.message
            msg_dict = self._message_to_dict(assistant_msg)
            messages.append(msg_dict)

            if choice.finish_reason == "tool_calls" or assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls or []:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    yield AgentTurnEvent(
                        type="tool_call",
                        tool_name=tc.function.name,
                        tool_args=args,
                        turn=turn,
                    )

                    result = self._execute_tool(tc.function.name, args)

                    yield AgentTurnEvent(
                        type="tool_result",
                        content=str(result),
                        tool_name=tc.function.name,
                        turn=turn,
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tc.id,
                        }
                    )
                continue

            content = assistant_msg.content or ""
            yield AgentTurnEvent(type="text", content=content, turn=turn)
            yield AgentTurnEvent(type="done", content=content, turn=turn)
            return

        yield AgentTurnEvent(
            type="done",
            content="Max turns reached.",
            turn=self.max_turns,
        )

    def _build_initial_messages(
        self, prompt: str, context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Build the initial message array for the agent loop."""
        messages: List[Dict[str, Any]] = []

        # System prompt
        system = self.system_prompt
        if context:
            system += f"\n\nAdditional context:\n{context}"
        messages.append({"role": "system", "content": system})

        # Add memory messages (previous conversation)
        memory_msgs = self.memory.get_messages()
        for msg in memory_msgs:
            if msg.get("role") != "system":
                messages.append(msg)

        # User prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def _call_llm(self, messages: List[Dict[str, Any]]) -> ChatResponse:
        """Call the LLM with current messages and tool definitions."""
        kwargs: Dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        tools = self._registry.to_openai_tools() if len(self._registry) > 0 else None

        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            tools=tools,
            **kwargs,
        )

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name with the given arguments."""
        tool_def = self._registry.get(name)
        if tool_def is None:
            return f"Error: Unknown tool '{name}'"

        try:
            result = tool_def.execute(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {e}"

    def _message_to_dict(self, msg: Any) -> Dict[str, Any]:
        """Convert a ChatMessage to a dict for the messages array."""
        d: Dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            d["content"] = msg.content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        return d
