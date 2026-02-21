"""Tests for shell execution tool."""

from unittest.mock import MagicMock, patch

import pytest

from meganova.agents.tools.shell import shell_tool, _execute_shell
from meganova.agents.tools.base import ToolDefinition


class TestShellTool:
    def test_is_tool_definition(self):
        assert isinstance(shell_tool, ToolDefinition)
        assert shell_tool.name == "execute_shell"

    def test_has_parameters(self):
        assert "command" in shell_tool.parameters["properties"]
        assert "command" in shell_tool.parameters["required"]

    @patch("meganova.agents.tools.shell.subprocess.run")
    def test_successful_command(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="hello world", stderr="", returncode=0
        )
        result = _execute_shell("echo hello world")
        assert result == "hello world"

    @patch("meganova.agents.tools.shell.subprocess.run")
    def test_command_with_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="", stderr="not found", returncode=1
        )
        result = _execute_shell("bad_command")
        assert "STDERR: not found" in result
        assert "Exit code: 1" in result

    @patch("meganova.agents.tools.shell.subprocess.run")
    def test_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        result = _execute_shell("true")
        assert result == "(no output)"

    @patch("meganova.agents.tools.shell.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
        result = _execute_shell("sleep 100", timeout=30)
        assert "timed out" in result

    @patch("meganova.agents.tools.shell.subprocess.run")
    def test_exception(self, mock_run):
        mock_run.side_effect = OSError("permission denied")
        result = _execute_shell("restricted_cmd")
        assert "Error:" in result

    @patch("meganova.agents.tools.shell.subprocess.run")
    def test_working_dir(self, mock_run):
        mock_run.return_value = MagicMock(stdout="ok", stderr="", returncode=0)
        _execute_shell("ls", working_dir="/tmp")
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["cwd"] == "/tmp"

    @patch("meganova.agents.tools.shell.subprocess.run")
    def test_custom_timeout(self, mock_run):
        mock_run.return_value = MagicMock(stdout="ok", stderr="", returncode=0)
        _execute_shell("ls", timeout=60)
        assert mock_run.call_args.kwargs["timeout"] == 60
