"""Base sandbox provider abstraction.

All sandbox providers (Docker, E2B, etc.) implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SandboxStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SandboxConfig:
    """Configuration for a sandbox instance."""

    image: str = "python:3.11-slim"
    memory_mb: int = 512
    cpu_count: int = 1
    timeout_seconds: int = 1800  # 30 minutes
    network_enabled: bool = False
    read_only_root: bool = True
    working_dir: str = "/workspace"
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Result from executing a command in a sandbox."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False


class SandboxProvider(ABC):
    """Abstract base class for sandbox providers."""

    provider_id: str = ""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._status = SandboxStatus.CREATED

    @abstractmethod
    def start(self) -> None:
        """Start the sandbox environment."""

    @abstractmethod
    def stop(self) -> None:
        """Stop and clean up the sandbox."""

    @abstractmethod
    def execute(self, command: str, timeout: Optional[int] = None) -> SandboxResult:
        """Execute a command in the sandbox."""

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Write a file inside the sandbox."""

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the sandbox is currently running."""

    @property
    def status(self) -> SandboxStatus:
        return self._status

    def __enter__(self) -> "SandboxProvider":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
