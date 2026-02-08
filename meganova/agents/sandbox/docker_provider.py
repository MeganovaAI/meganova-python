"""Docker sandbox provider.

Runs agent commands inside isolated Docker containers with configurable
security settings (network isolation, resource limits, read-only root).
"""

import uuid
from typing import Any, Dict, Optional

from .base import SandboxConfig, SandboxProvider, SandboxResult, SandboxStatus


class DockerSandbox(SandboxProvider):
    """Sandbox using Docker containers.

    Requires: pip install meganova[agents]  (docker package)

    Usage:
        from meganova.agents.sandbox.docker_provider import DockerSandbox

        config = SandboxConfig(image="python:3.11-slim", memory_mb=512)
        with DockerSandbox(config) as sandbox:
            result = sandbox.execute("python -c 'print(1+1)'")
            print(result.stdout)  # "2"
    """

    provider_id = "docker"

    def __init__(self, config: Optional[SandboxConfig] = None):
        super().__init__(config)
        self._container = None
        self._container_name = f"nova-sandbox-{uuid.uuid4().hex[:12]}"

    def start(self) -> None:
        """Start a Docker container."""
        try:
            import docker
        except ImportError:
            raise ImportError(
                "Docker package not installed. Install with: pip install meganova[agents]"
            )

        client = docker.from_env()

        # Build container options with security defaults
        run_kwargs: Dict[str, Any] = {
            "image": self.config.image,
            "name": self._container_name,
            "detach": True,
            "tty": True,
            "working_dir": self.config.working_dir,
            "mem_limit": f"{self.config.memory_mb}m",
            "nano_cpus": int(self.config.cpu_count * 1e9),
            "pids_limit": 100,
            "user": "1000:1000",
            "environment": self.config.environment,
        }

        if not self.config.network_enabled:
            run_kwargs["network_mode"] = "none"

        if self.config.read_only_root:
            run_kwargs["read_only"] = True
            # Need tmpfs for /tmp and /workspace
            run_kwargs["tmpfs"] = {
                "/tmp": "size=100m",
                self.config.working_dir: "size=200m",
            }

        # Security options
        run_kwargs["cap_drop"] = ["ALL"]
        run_kwargs["security_opt"] = ["no-new-privileges"]

        self._container = client.containers.run(
            command="sleep infinity",
            **run_kwargs,
        )
        self._status = SandboxStatus.RUNNING

    def stop(self) -> None:
        """Stop and remove the container."""
        if self._container:
            try:
                self._container.stop(timeout=5)
            except Exception:
                try:
                    self._container.kill()
                except Exception:
                    pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None
        self._status = SandboxStatus.STOPPED

    def execute(self, command: str, timeout: Optional[int] = None) -> SandboxResult:
        """Execute a command inside the container."""
        if not self._container:
            return SandboxResult(stderr="Sandbox not started", exit_code=1)

        exec_timeout = timeout or self.config.timeout_seconds

        try:
            exit_code, output = self._container.exec_run(
                cmd=["sh", "-c", command],
                workdir=self.config.working_dir,
                user="1000:1000",
                demux=True,
            )

            stdout = ""
            stderr = ""
            if output:
                if isinstance(output, tuple):
                    stdout = (output[0] or b"").decode("utf-8", errors="replace")
                    stderr = (output[1] or b"").decode("utf-8", errors="replace")
                else:
                    stdout = output.decode("utf-8", errors="replace")

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )

        except Exception as e:
            return SandboxResult(
                stderr=f"Execution error: {e}",
                exit_code=1,
            )

    def write_file(self, path: str, content: str) -> None:
        """Write a file inside the container."""
        if not self._container:
            raise RuntimeError("Sandbox not started")

        import io
        import tarfile

        # Create a tar archive with the file
        data = content.encode("utf-8")
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            info = tarfile.TarInfo(name=path.lstrip("/"))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        stream.seek(0)

        self._container.put_archive("/", stream)

    def read_file(self, path: str) -> str:
        """Read a file from the container."""
        result = self.execute(f"cat {path}")
        if result.exit_code != 0:
            raise FileNotFoundError(f"Cannot read {path}: {result.stderr}")
        return result.stdout

    def is_running(self) -> bool:
        """Check if the container is running."""
        if not self._container:
            return False
        try:
            self._container.reload()
            return self._container.status == "running"
        except Exception:
            return False
