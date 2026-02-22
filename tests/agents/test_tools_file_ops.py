"""Tests for file operation tools."""

import os

import pytest

from meganova.agents.tools.file_ops import (
    _edit_file,
    _list_directory,
    _read_file,
    _write_file,
    edit_file_tool,
    list_directory_tool,
    read_file_tool,
    write_file_tool,
)
from meganova.agents.tools.base import ToolDefinition


class TestReadFile:
    def test_is_tool_definition(self):
        assert isinstance(read_file_tool, ToolDefinition)
        assert read_file_tool.name == "read_file"

    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = _read_file(str(f))
        assert result == "hello world"

    def test_read_nonexistent_file(self):
        result = _read_file("/nonexistent/path/file.txt")
        assert "Error reading" in result

    def test_max_lines(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1000)))
        result = _read_file(str(f), max_lines=10)
        assert "more lines" in result

    def test_tilde_expansion(self, tmp_path):
        # Just verify the function calls expanduser
        result = _read_file("~/nonexistent_test_file_xyz")
        assert "Error reading" in result


class TestWriteFile:
    def test_is_tool_definition(self):
        assert isinstance(write_file_tool, ToolDefinition)
        assert write_file_tool.name == "write_file"

    def test_write_new_file(self, tmp_path):
        path = str(tmp_path / "output.txt")
        result = _write_file(path, "test content")
        assert "Written" in result
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "test content"

    def test_creates_directories(self, tmp_path):
        path = str(tmp_path / "subdir" / "nested" / "file.txt")
        result = _write_file(path, "nested content")
        assert "Written" in result
        assert os.path.exists(path)

    def test_overwrites_existing(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("old")
        _write_file(str(f), "new")
        assert f.read_text() == "new"


class TestEditFile:
    def test_is_tool_definition(self):
        assert isinstance(edit_file_tool, ToolDefinition)
        assert edit_file_tool.name == "edit_file"

    def test_replace_text(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = _edit_file(str(f), "world", "universe")
        assert "replaced text successfully" in result
        assert f.read_text() == "hello universe"

    def test_old_text_not_found(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = _edit_file(str(f), "missing", "replacement")
        assert "not found" in result

    def test_replaces_first_occurrence_only(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("aaa bbb aaa")
        _edit_file(str(f), "aaa", "ccc")
        assert f.read_text() == "ccc bbb aaa"

    def test_nonexistent_file(self):
        result = _edit_file("/nonexistent/file.txt", "old", "new")
        assert "Error editing" in result


class TestListDirectory:
    def test_is_tool_definition(self):
        assert isinstance(list_directory_tool, ToolDefinition)
        assert list_directory_tool.name == "list_directory"

    def test_list_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("test")
        (tmp_path / "subdir").mkdir()
        result = _list_directory(str(tmp_path))
        assert "f file.txt" in result
        assert "d subdir" in result

    def test_max_entries(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("")
        result = _list_directory(str(tmp_path), max_entries=3)
        assert "more" in result

    def test_nonexistent_directory(self):
        result = _list_directory("/nonexistent/path")
        assert "Error listing" in result

    def test_sorted_output(self, tmp_path):
        (tmp_path / "c.txt").write_text("")
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = _list_directory(str(tmp_path))
        lines = result.strip().split("\n")
        names = [l.split(" ", 1)[1] for l in lines]
        assert names == sorted(names)
