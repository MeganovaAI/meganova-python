"""E2B Firecracker sandbox provider.

Uses E2B's managed Firecracker microVMs for production-grade isolation.
Requires: pip install meganova[sandbox]  (e2b package)
"""

from typing import Optional

from .base import SandboxConfig, SandboxProvider, SandboxResult, SandboxStatus


class E2BSandbox(SandboxProvider):
    """Sandbox using E2B Firecracker microVMs.

    Requires:
        pip install meganova[sandbox]
        E2B_API_KEY environment variable

    Usage:
        from meganova.agents.sandbox.e2b_provider import E2BSandbox

        config = SandboxConfig(timeout_seconds=300)
        with E2BSandbox(config, api_key="e2b_...") as sandbox:
            result = sandbox.execute("python -c 'print(1+1)'")
            print(result.stdout)  # "2"
    """

    provider_id = "e2b"

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        api_key: Optional[str] = None,
        template: str = "base",
    ):
        super().__init__(config)
        self._api_key = api_key
        self._template = template
        self._sandbox = None

    def start(self) -> None:
        """Start an E2B sandbox."""
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            raise ImportError(
                "E2B package not installed. Install with: pip install meganova[sandbox]"
            )

        kwargs = {
            "template": self._template,
            "timeout": self.config.timeout_seconds,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key

        self._sandbox = Sandbox(**kwargs)
        self._status = SandboxStatus.RUNNING

    def stop(self) -> None:
        """Stop the E2B sandbox."""
        if self._sandbox:
            try:
                self._sandbox.kill()
            except Exception:
                pass
            self._sandbox = None
        self._status = SandboxStatus.STOPPED

    def execute(self, command: str, timeout: Optional[int] = None) -> SandboxResult:
        """Execute a command in the E2B sandbox."""
        if not self._sandbox:
            return SandboxResult(stderr="Sandbox not started", exit_code=1)

        try:
            result = self._sandbox.commands.run(
                command,
                timeout=timeout or self.config.timeout_seconds,
            )
            return SandboxResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
            )
        except Exception as e:
            return SandboxResult(stderr=f"E2B error: {e}", exit_code=1)

    def write_file(self, path: str, content: str) -> None:
        """Write a file in the E2B sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")
        self._sandbox.files.write(path, content)

    def read_file(self, path: str) -> str:
        """Read a file from the E2B sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")
        return self._sandbox.files.read(path)

    def is_running(self) -> bool:
        """Check if the sandbox is running."""
        return self._sandbox is not None and self._status == SandboxStatus.RUNNING
