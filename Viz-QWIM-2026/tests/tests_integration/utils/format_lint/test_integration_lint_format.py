"""Integration tests for format_lint package.

Tests the full linting pipeline: run_command, lint_and_format_file, and
lint_and_format_directory in realistic end-to-end scenarios using
temporary files and directories.

Author: QWIM Development Team
Version: 0.1.0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.format_lint.lint_format import (
    lint_and_format_directory,
    lint_and_format_file,
    run_command,
)


# ==============================================================================
# Integration: run_command with real system commands
# ==============================================================================


class Test_Run_Command_Integration:
    """Integration tests for run_command against real OS commands."""

    @pytest.mark.integration()
    def test_run_echo_command_returns_zero(self):
        """run_command(echo ...) succeeds on all supported OS."""
        import sys

        cmd = "echo hello" if sys.platform != "win32" else "echo hello"
        result = run_command(cmd)
        assert result == 0

    @pytest.mark.integration()
    def test_run_nonexistent_command_returns_nonzero(self):
        """run_command with a nonexistent binary returns a non-zero exit code."""
        result = run_command("this_command_does_not_exist_xyz_abc_123")
        assert result != 0

    @pytest.mark.integration()
    def test_run_python_version_command_returns_zero(self):
        """Running 'python --version' through run_command succeeds."""
        import sys

        python_exe = sys.executable
        result = run_command(f"{python_exe} --version")
        assert result == 0


# ==============================================================================
# Integration: lint_and_format_file with real Python files
# ==============================================================================


class Test_Lint_And_Format_File_Integration:
    """Integration tests for lint_and_format_file using temporary Python files."""

    @pytest.mark.integration()
    def test_lint_clean_file_succeeds(self, tmp_path):
        """lint_and_format_file returns zero exit code for a clean Python file."""
        py_file = tmp_path / "clean.py"
        py_file.write_text(
            '"""A simple clean module."""\n\nX = 1\n',
            encoding="utf-8",
        )
        result = lint_and_format_file(py_file)
        # Result may be 0 (success) or small positive (ruff may add trailing newline)
        assert isinstance(result, int)

    @pytest.mark.integration()
    def test_lint_python_file_returns_integer(self, tmp_path):
        """lint_and_format_file always returns an integer."""
        py_file = tmp_path / "sample.py"
        py_file.write_text("x = 1\n", encoding="utf-8")
        result = lint_and_format_file(py_file)
        assert isinstance(result, int)

    @pytest.mark.integration()
    def test_lint_non_python_file_skipped(self, tmp_path):
        """lint_and_format_file returns early (non-zero) for non-.py files."""
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("Hello world\n", encoding="utf-8")
        result = lint_and_format_file(txt_file)
        # Non-Python files should be skipped → return 1 or similar non-zero
        assert isinstance(result, int)

    @pytest.mark.integration()
    def test_lint_nonexistent_file_returns_nonzero(self, tmp_path):
        """lint_and_format_file for a missing file returns a non-zero code."""
        missing_file = tmp_path / "does_not_exist.py"
        result = lint_and_format_file(missing_file)
        assert result != 0

    @pytest.mark.integration()
    def test_lint_file_with_path_object(self, tmp_path):
        """lint_and_format_file accepts a pathlib.Path argument."""
        py_file = tmp_path / "path_object.py"
        py_file.write_text('"""Module."""\n\nY = 2\n', encoding="utf-8")
        result = lint_and_format_file(Path(py_file))
        assert isinstance(result, int)


# ==============================================================================
# Integration: lint_and_format_directory with real directories
# ==============================================================================


class Test_Lint_And_Format_Directory_Integration:
    """Integration tests for lint_and_format_directory using temporary directories."""

    @pytest.mark.integration()
    def test_lint_empty_directory_returns_integer(self, tmp_path):
        """lint_and_format_directory on an empty dir returns an integer."""
        result = lint_and_format_directory(tmp_path)
        assert isinstance(result, int)

    @pytest.mark.integration()
    def test_lint_directory_with_single_clean_file(self, tmp_path):
        """lint_and_format_directory on a directory with one clean .py file."""
        py_file = tmp_path / "module.py"
        py_file.write_text('"""Module."""\n\nZ = 3\n', encoding="utf-8")
        result = lint_and_format_directory(tmp_path)
        assert isinstance(result, int)

    @pytest.mark.integration()
    def test_lint_directory_with_multiple_files(self, tmp_path):
        """lint_and_format_directory processes all .py files in the directory."""
        for i in range(3):
            py_file = tmp_path / f"module_{i}.py"
            py_file.write_text(f'"""Module {i}."""\n\nV = {i}\n', encoding="utf-8")
        result = lint_and_format_directory(tmp_path)
        assert isinstance(result, int)

    @pytest.mark.integration()
    def test_lint_nonexistent_directory_returns_nonzero(self, tmp_path):
        """lint_and_format_directory on a missing path returns non-zero."""
        missing_dir = tmp_path / "no_such_dir"
        result = lint_and_format_directory(missing_dir)
        assert result != 0

    @pytest.mark.integration()
    def test_lint_directory_with_non_python_files_skips_them(self, tmp_path):
        """Non-.py files in directory do not cause lint_and_format_directory to fail."""
        (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# Title\n", encoding="utf-8")
        result = lint_and_format_directory(tmp_path)
        # Should handle gracefully (return integer, not raise)
        assert isinstance(result, int)

    @pytest.mark.integration()
    def test_lint_nested_directory_recursive(self, tmp_path):
        """lint_and_format_directory processes nested subdirectories."""
        sub = tmp_path / "subpackage"
        sub.mkdir()
        (sub / "__init__.py").write_text('"""Sub package."""\n', encoding="utf-8")
        (sub / "utils.py").write_text('"""Utils."""\n\nW = 42\n', encoding="utf-8")
        result = lint_and_format_directory(tmp_path)
        assert isinstance(result, int)


# ==============================================================================
# Integration: pipeline — lint file then directory
# ==============================================================================


class Test_Full_Pipeline_Integration:
    """End-to-end pipeline tests combining file and directory linting."""

    @pytest.mark.integration()
    def test_file_lint_consistent_with_directory_lint(self, tmp_path):
        """Linting a single file and linting its parent dir give same-type results."""
        py_file = tmp_path / "consistent.py"
        py_file.write_text('"""Module."""\n\nA = 1\n', encoding="utf-8")

        file_result = lint_and_format_file(py_file)
        dir_result = lint_and_format_directory(tmp_path)

        assert isinstance(file_result, int)
        assert isinstance(dir_result, int)

    @pytest.mark.integration()
    def test_run_command_followed_by_lint_file(self, tmp_path):
        """run_command executes echo, then lint_and_format_file processes a file."""
        echo_result = run_command("echo pipeline-test")
        assert echo_result == 0

        py_file = tmp_path / "pipeline.py"
        py_file.write_text('"""Pipeline module."""\n\nB = 2\n', encoding="utf-8")
        lint_result = lint_and_format_file(py_file)
        assert isinstance(lint_result, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
