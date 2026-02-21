"""Tests for sandbox system."""

from unittest.mock import MagicMock, patch

import pytest

from meganova.agents.sandbox.base import (
    SandboxConfig,
    SandboxProvider,
    SandboxResult,
    SandboxStatus,
)


class TestSandboxStatus:
    def test_values(self):
        assert SandboxStatus.CREATED == "created"
        assert SandboxStatus.RUNNING == "running"
        assert SandboxStatus.STOPPED == "stopped"
        assert SandboxStatus.ERROR == "error"


class TestSandboxConfig:
    def test_defaults(self):
        cfg = SandboxConfig()
        assert cfg.image == "python:3.11-slim"
        assert cfg.memory_mb == 512
        assert cfg.cpu_count == 1
        assert cfg.timeout_seconds == 1800
        assert cfg.network_enabled is False
        assert cfg.read_only_root is True
        assert cfg.working_dir == "/workspace"
        assert cfg.environment == {}

    def test_custom_config(self):
        cfg = SandboxConfig(
            image="node:18", memory_mb=1024, cpu_count=2,
            timeout_seconds=300, network_enabled=True,
            working_dir="/app", environment={"NODE_ENV": "test"},
        )
        assert cfg.image == "node:18"
        assert cfg.memory_mb == 1024
        assert cfg.network_enabled is True
        assert cfg.environment["NODE_ENV"] == "test"


class TestSandboxResult:
    def test_defaults(self):
        result = SandboxResult()
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_custom(self):
        result = SandboxResult(stdout="hello", exit_code=1, timed_out=True)
        assert result.stdout == "hello"
        assert result.exit_code == 1
        assert result.timed_out is True


class TestSandboxProvider:
    def test_default_status(self):
        """Concrete subclass to test base behavior."""
        class FakeSandbox(SandboxProvider):
            provider_id = "fake"
            def start(self): self._status = SandboxStatus.RUNNING
            def stop(self): self._status = SandboxStatus.STOPPED
            def execute(self, command, timeout=None): return SandboxResult()
            def write_file(self, path, content): pass
            def read_file(self, path): return ""
            def is_running(self): return self._status == SandboxStatus.RUNNING

        sb = FakeSandbox()
        assert sb.status == SandboxStatus.CREATED

    def test_context_manager(self):
        class FakeSandbox(SandboxProvider):
            provider_id = "fake"
            started = False
            stopped = False
            def start(self): self.started = True; self._status = SandboxStatus.RUNNING
            def stop(self): self.stopped = True; self._status = SandboxStatus.STOPPED
            def execute(self, command, timeout=None): return SandboxResult()
            def write_file(self, path, content): pass
            def read_file(self, path): return ""
            def is_running(self): return self._status == SandboxStatus.RUNNING

        sb = FakeSandbox()
        with sb as s:
            assert s.started
        assert s.stopped

    def test_default_config(self):
        class FakeSandbox(SandboxProvider):
            provider_id = "fake"
            def start(self): pass
            def stop(self): pass
            def execute(self, command, timeout=None): return SandboxResult()
            def write_file(self, path, content): pass
            def read_file(self, path): return ""
            def is_running(self): return False

        sb = FakeSandbox()
        assert isinstance(sb.config, SandboxConfig)

    def test_custom_config(self):
        class FakeSandbox(SandboxProvider):
            provider_id = "fake"
            def start(self): pass
            def stop(self): pass
            def execute(self, command, timeout=None): return SandboxResult()
            def write_file(self, path, content): pass
            def read_file(self, path): return ""
            def is_running(self): return False

        cfg = SandboxConfig(image="custom:latest")
        sb = FakeSandbox(config=cfg)
        assert sb.config.image == "custom:latest"


class TestDockerSandbox:
    def test_provider_id(self):
        from meganova.agents.sandbox.docker_provider import DockerSandbox
        assert DockerSandbox.provider_id == "docker"

    def test_init_creates_container_name(self):
        from meganova.agents.sandbox.docker_provider import DockerSandbox
        sb = DockerSandbox()
        assert sb._container_name.startswith("nova-sandbox-")

    def test_execute_without_start(self):
        from meganova.agents.sandbox.docker_provider import DockerSandbox
        sb = DockerSandbox()
        result = sb.execute("echo hi")
        assert result.exit_code == 1
        assert "not started" in result.stderr

    def test_stop_without_start(self):
        from meganova.agents.sandbox.docker_provider import DockerSandbox
        sb = DockerSandbox()
        sb.stop()  # Should not raise
        assert sb.status == SandboxStatus.STOPPED

    def test_is_running_without_start(self):
        from meganova.agents.sandbox.docker_provider import DockerSandbox
        sb = DockerSandbox()
        assert not sb.is_running()


class TestE2BSandbox:
    def test_provider_id(self):
        from meganova.agents.sandbox.e2b_provider import E2BSandbox
        assert E2BSandbox.provider_id == "e2b"

    def test_init_stores_api_key(self):
        from meganova.agents.sandbox.e2b_provider import E2BSandbox
        sb = E2BSandbox(api_key="e2b_test_key", template="python")
        assert sb._api_key == "e2b_test_key"
        assert sb._template == "python"

    def test_execute_without_start(self):
        from meganova.agents.sandbox.e2b_provider import E2BSandbox
        sb = E2BSandbox()
        result = sb.execute("echo hi")
        assert result.exit_code == 1

    def test_stop_without_start(self):
        from meganova.agents.sandbox.e2b_provider import E2BSandbox
        sb = E2BSandbox()
        sb.stop()
        assert sb.status == SandboxStatus.STOPPED

    def test_is_running_without_start(self):
        from meganova.agents.sandbox.e2b_provider import E2BSandbox
        sb = E2BSandbox()
        assert not sb.is_running()
