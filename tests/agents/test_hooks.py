"""Tests for hook system."""

import pytest

from meganova.agents.hooks import (
    Hook,
    HookContext,
    HookManager,
    HookResult,
    HookType,
    on_error,
    post_model_call,
    post_tool_use,
    pre_model_call,
    pre_tool_use,
)


class TestHookType:
    def test_all_types(self):
        types = list(HookType)
        assert len(types) == 6
        assert HookType.PRE_TOOL_USE in types
        assert HookType.POST_TOOL_USE in types
        assert HookType.PRE_MODEL_CALL in types
        assert HookType.POST_MODEL_CALL in types
        assert HookType.ON_ERROR in types
        assert HookType.ON_COMPLETE in types

    def test_string_values(self):
        assert HookType.PRE_TOOL_USE == "pre_tool_use"


class TestHookContext:
    def test_minimal(self):
        ctx = HookContext(agent_name="test", turn=0)
        assert ctx.agent_name == "test"
        assert ctx.tool_name is None

    def test_full(self):
        ctx = HookContext(
            agent_name="test", turn=1,
            tool_name="search", tool_args={"q": "hi"},
            tool_result="results", messages=[],
            response="Hello", error=None,
        )
        assert ctx.tool_name == "search"
        assert ctx.tool_args == {"q": "hi"}


class TestHookResult:
    def test_defaults(self):
        r = HookResult()
        assert r.allow is True
        assert r.modified_args is None
        assert r.reason is None

    def test_deny(self):
        r = HookResult(allow=False, reason="blocked")
        assert not r.allow
        assert r.reason == "blocked"


class TestHook:
    def test_call(self):
        called = []
        def fn(ctx):
            called.append(ctx)
            return HookResult(allow=True)

        hook = Hook(HookType.PRE_TOOL_USE, fn)
        ctx = HookContext(agent_name="test", turn=0)
        result = hook(ctx)
        assert len(called) == 1
        assert result.allow is True

    def test_attributes(self):
        hook = Hook(HookType.ON_ERROR, lambda ctx: None, priority=5)
        assert hook.hook_type == HookType.ON_ERROR
        assert hook.priority == 5


class TestHookManager:
    def test_register_and_run(self):
        mgr = HookManager()
        calls = []
        mgr.register(HookType.PRE_TOOL_USE, lambda ctx: calls.append("called"))

        ctx = HookContext(agent_name="test", turn=0)
        mgr.run(HookType.PRE_TOOL_USE, ctx)
        assert len(calls) == 1

    def test_priority_ordering(self):
        mgr = HookManager()
        calls = []
        mgr.register(HookType.PRE_TOOL_USE, lambda ctx: calls.append("low"), priority=1)
        mgr.register(HookType.PRE_TOOL_USE, lambda ctx: calls.append("high"), priority=10)

        mgr.run(HookType.PRE_TOOL_USE, HookContext(agent_name="test", turn=0))
        assert calls == ["high", "low"]

    def test_deny_stops_execution(self):
        mgr = HookManager()
        calls = []

        def deny(ctx):
            calls.append("deny")
            return HookResult(allow=False, reason="blocked")

        def after(ctx):
            calls.append("after")
            return HookResult()

        mgr.register(HookType.PRE_TOOL_USE, deny, priority=10)
        mgr.register(HookType.PRE_TOOL_USE, after, priority=1)

        result = mgr.run(HookType.PRE_TOOL_USE, HookContext(agent_name="test", turn=0))
        assert not result.allow
        assert result.reason == "blocked"
        assert "after" not in calls

    def test_modified_args_propagated(self):
        mgr = HookManager()
        mgr.register(
            HookType.PRE_TOOL_USE,
            lambda ctx: HookResult(modified_args={"x": "modified"}),
        )

        result = mgr.run(HookType.PRE_TOOL_USE, HookContext(agent_name="test", turn=0))
        assert result.modified_args == {"x": "modified"}

    def test_modified_messages_propagated(self):
        mgr = HookManager()
        new_msgs = [{"role": "user", "content": "modified"}]
        mgr.register(
            HookType.PRE_MODEL_CALL,
            lambda ctx: HookResult(modified_messages=new_msgs),
        )

        result = mgr.run(HookType.PRE_MODEL_CALL, HookContext(agent_name="test", turn=0))
        assert result.modified_messages == new_msgs

    def test_has_hooks(self):
        mgr = HookManager()
        assert not mgr.has_hooks(HookType.PRE_TOOL_USE)
        mgr.register(HookType.PRE_TOOL_USE, lambda ctx: None)
        assert mgr.has_hooks(HookType.PRE_TOOL_USE)

    def test_no_hooks_returns_allow(self):
        mgr = HookManager()
        result = mgr.run(HookType.PRE_TOOL_USE, HookContext(agent_name="test", turn=0))
        assert result.allow is True

    def test_hook_exception_continues(self):
        mgr = HookManager()
        calls = []

        def bad_hook(ctx):
            raise RuntimeError("boom")

        def good_hook(ctx):
            calls.append("good")
            return HookResult()

        mgr.register(HookType.PRE_TOOL_USE, bad_hook, priority=10)
        mgr.register(HookType.PRE_TOOL_USE, good_hook, priority=1)

        result = mgr.run(HookType.PRE_TOOL_USE, HookContext(agent_name="test", turn=0))
        assert result.allow is True
        assert "good" in calls

    def test_none_return_continues(self):
        mgr = HookManager()
        mgr.register(HookType.PRE_TOOL_USE, lambda ctx: None)

        result = mgr.run(HookType.PRE_TOOL_USE, HookContext(agent_name="test", turn=0))
        assert result.allow is True

    def test_different_hook_types_independent(self):
        mgr = HookManager()
        mgr.register(HookType.PRE_TOOL_USE, lambda ctx: None)
        assert mgr.has_hooks(HookType.PRE_TOOL_USE)
        assert not mgr.has_hooks(HookType.POST_TOOL_USE)


class TestHookDecorators:
    def test_pre_tool_use(self):
        @pre_tool_use(priority=5)
        def my_hook(ctx):
            return HookResult()

        assert isinstance(my_hook, Hook)
        assert my_hook.hook_type == HookType.PRE_TOOL_USE
        assert my_hook.priority == 5

    def test_post_tool_use(self):
        @post_tool_use()
        def my_hook(ctx):
            return HookResult()

        assert my_hook.hook_type == HookType.POST_TOOL_USE

    def test_pre_model_call(self):
        @pre_model_call()
        def my_hook(ctx):
            return HookResult()

        assert my_hook.hook_type == HookType.PRE_MODEL_CALL

    def test_post_model_call(self):
        @post_model_call()
        def my_hook(ctx):
            return HookResult()

        assert my_hook.hook_type == HookType.POST_MODEL_CALL

    def test_on_error(self):
        @on_error()
        def my_hook(ctx):
            return HookResult()

        assert my_hook.hook_type == HookType.ON_ERROR

    def test_decorated_hook_callable(self):
        @pre_tool_use()
        def my_hook(ctx):
            return HookResult(modified_args={"test": True})

        ctx = HookContext(agent_name="test", turn=0)
        result = my_hook(ctx)
        assert result.modified_args == {"test": True}

    def test_default_priority(self):
        @pre_tool_use()
        def my_hook(ctx):
            pass

        assert my_hook.priority == 0
