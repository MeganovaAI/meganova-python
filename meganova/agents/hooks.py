"""Hook system for agent lifecycle events.

Hooks allow intercepting and modifying agent behavior at key points:
- Before/after tool execution (credit checks, audit logging)
- Before/after model calls (prompt modification, response filtering)
- On agent completion or error
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class HookType(str, Enum):
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    PRE_MODEL_CALL = "pre_model_call"
    POST_MODEL_CALL = "post_model_call"
    ON_ERROR = "on_error"
    ON_COMPLETE = "on_complete"


@dataclass
class HookContext:
    """Context passed to hook functions."""

    agent_name: str
    turn: int
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    messages: Optional[List[Dict[str, Any]]] = None
    response: Optional[Any] = None
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HookResult:
    """Result from a hook execution."""

    allow: bool = True
    modified_args: Optional[Dict[str, Any]] = None
    modified_messages: Optional[List[Dict[str, Any]]] = None
    reason: Optional[str] = None


class Hook:
    """A registered hook function."""

    def __init__(
        self,
        hook_type: HookType,
        func: Callable,
        priority: int = 0,
    ):
        self.hook_type = hook_type
        self.func = func
        self.priority = priority

    def __call__(self, context: HookContext) -> Optional[HookResult]:
        return self.func(context)


class HookManager:
    """Manages hook registration and execution."""

    def __init__(self) -> None:
        self._hooks: Dict[HookType, List[Hook]] = {ht: [] for ht in HookType}

    def register(
        self,
        hook_type: HookType,
        func: Callable,
        priority: int = 0,
    ) -> None:
        hook = Hook(hook_type, func, priority)
        self._hooks[hook_type].append(hook)
        self._hooks[hook_type].sort(key=lambda h: h.priority, reverse=True)

    def run(self, hook_type: HookType, context: HookContext) -> HookResult:
        """Run all hooks of the given type. Returns combined result."""
        result = HookResult(allow=True)

        for hook in self._hooks[hook_type]:
            try:
                hook_result = hook(context)
                if hook_result is not None:
                    if not hook_result.allow:
                        return hook_result
                    if hook_result.modified_args is not None:
                        result.modified_args = hook_result.modified_args
                    if hook_result.modified_messages is not None:
                        result.modified_messages = hook_result.modified_messages
            except Exception:
                continue

        return result

    def has_hooks(self, hook_type: HookType) -> bool:
        return len(self._hooks[hook_type]) > 0


def pre_tool_use(priority: int = 0) -> Callable:
    """Decorator to register a pre-tool-use hook."""

    def decorator(func: Callable) -> Hook:
        return Hook(HookType.PRE_TOOL_USE, func, priority)

    return decorator


def post_tool_use(priority: int = 0) -> Callable:
    """Decorator to register a post-tool-use hook."""

    def decorator(func: Callable) -> Hook:
        return Hook(HookType.POST_TOOL_USE, func, priority)

    return decorator


def pre_model_call(priority: int = 0) -> Callable:
    """Decorator to register a pre-model-call hook."""

    def decorator(func: Callable) -> Hook:
        return Hook(HookType.PRE_MODEL_CALL, func, priority)

    return decorator


def post_model_call(priority: int = 0) -> Callable:
    """Decorator to register a post-model-call hook."""

    def decorator(func: Callable) -> Hook:
        return Hook(HookType.POST_MODEL_CALL, func, priority)

    return decorator


def on_error(priority: int = 0) -> Callable:
    """Decorator to register an on-error hook."""

    def decorator(func: Callable) -> Hook:
        return Hook(HookType.ON_ERROR, func, priority)

    return decorator
