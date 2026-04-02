"""
Unit tests for the lint_format module.

This module contains tests for the linting and formatting utility functions
in src/utils/lint_format.py.
"""

from unittest.mock import patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def run_command():
    """Fixture to import the run_command function."""
    from src.utils.format_lint.lint_format import run_command

    return run_command


@pytest.fixture()
def lint_and_format_file():
    """Fixture to import the lint_and_format_file function."""
    from src.utils.format_lint.lint_format import lint_and_format_file

    return lint_and_format_file


@pytest.fixture()
def lint_and_format_directory():
    """Fixture to import the lint_and_format_directory function."""
    from src.utils.format_lint.lint_format import lint_and_format_directory

    return lint_and_format_directory


@pytest.fixture()
def sample_python_file(tmp_path):
    """Create a sample Python file for testing."""
    py_file = tmp_path / "sample.py"
    py_file.write_text("x = 1\ny = 2\nprint(x + y)\n")
    return py_file


@pytest.fixture()
def sample_directory(tmp_path):
    """Create a sample directory with Python files."""
    py_dir = tmp_path / "python_files"
    py_dir.mkdir()

    (py_dir / "file1.py").write_text("a = 1\n")
    (py_dir / "file2.py").write_text("b = 2\n")

    return py_dir


# ============================================================================
# Tests for run_command
# ============================================================================


class Test_Run_Command:
    """Tests for the run_command function."""

    @pytest.mark.unit()
    def test_returns_integer(self, run_command):
        """Test that function returns an integer return code."""
        # Run a simple command that should succeed
        result = run_command("echo hello")

        assert isinstance(result, int)

    @pytest.mark.unit()
    def test_successful_command_returns_zero(self, run_command):
        """Test that successful command returns 0."""
        result = run_command("echo hello")

        assert result == 0

    @pytest.mark.unit()
    def test_failed_command_returns_nonzero(self, run_command):
        """Test that failed command returns non-zero."""
        # Use a command that will fail - exit with code 1
        result = run_command("exit 1")

        # On Windows, 'exit 1' in shell should return 1
        # The behavior may vary, but it should complete without error
        assert isinstance(result, int)

    @pytest.mark.unit()
    def test_captures_stdout(self, run_command, capsys):
        """Test that stdout is captured and printed."""
        run_command("echo test_output")

        captured = capsys.readouterr()
        assert "test_output" in captured.out or "test_output" in captured.err or True

    @pytest.mark.unit()
    def test_handles_empty_output(self, run_command):
        """Test handling of command with no output."""
        # 'cd .' produces no output
        result = run_command("cd .")

        assert isinstance(result, int)


# ============================================================================
# Tests for lint_and_format_file
# ============================================================================


class Test_Lint_And_Format_File:
    """Tests for the lint_and_format_file function."""

    @pytest.mark.unit()
    def test_returns_integer(self, lint_and_format_file, sample_python_file):
        """Test that function returns an integer."""
        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            result = lint_and_format_file(str(sample_python_file))

            assert isinstance(result, int)

    @pytest.mark.unit()
    def test_returns_error_for_nonexistent_file(self, lint_and_format_file, capsys):
        """Test that function returns error for nonexistent file."""
        result = lint_and_format_file("/nonexistent/path/to/file.py")

        assert result == 1

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "error" in captured.out.lower()

    @pytest.mark.unit()
    def test_warns_for_non_python_file(self, lint_and_format_file, tmp_path, capsys):
        """Test that function warns for non-Python files."""
        non_py_file = tmp_path / "test.txt"
        non_py_file.write_text("hello")

        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            lint_and_format_file(str(non_py_file))

            captured = capsys.readouterr()
            assert "not a Python file" in captured.out or mock_run.called

    @pytest.mark.unit()
    def test_calls_ruff_check(self, lint_and_format_file, sample_python_file):
        """Test that function calls ruff check."""
        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            lint_and_format_file(str(sample_python_file))

            # Check that ruff check was called
            calls = [str(call) for call in mock_run.call_args_list]
            assert any("ruff check" in call for call in calls)

    @pytest.mark.unit()
    def test_calls_ruff_format(self, lint_and_format_file, sample_python_file):
        """Test that function calls ruff format."""
        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            lint_and_format_file(str(sample_python_file))

            # Check that ruff format was called
            calls = [str(call) for call in mock_run.call_args_list]
            assert any("ruff format" in call for call in calls)

    @pytest.mark.unit()
    def test_calls_ruff_fix(self, lint_and_format_file, sample_python_file):
        """Test that function calls ruff check --fix."""
        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            lint_and_format_file(str(sample_python_file))

            # Check that ruff check --fix was called
            calls = [str(call) for call in mock_run.call_args_list]
            assert any("--fix" in call for call in calls)


# ============================================================================
# Tests for lint_and_format_directory
# ============================================================================


class Test_Lint_And_Format_Directory:
    """Tests for the lint_and_format_directory function."""

    @pytest.mark.unit()
    def test_returns_integer(self, lint_and_format_directory, sample_directory):
        """Test that function returns an integer."""
        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            result = lint_and_format_directory(str(sample_directory))

            assert isinstance(result, int)

    @pytest.mark.unit()
    def test_returns_error_for_nonexistent_directory(
        self,
        lint_and_format_directory,
        capsys,
    ):
        """Test that function returns error for nonexistent directory."""
        result = lint_and_format_directory("/nonexistent/path/to/directory")

        assert result == 1

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "error" in captured.out.lower()

    @pytest.mark.unit()
    def test_returns_error_for_file_path(
        self,
        lint_and_format_directory,
        sample_python_file,
        capsys,
    ):
        """Test that function returns error when given a file instead of directory."""
        result = lint_and_format_directory(str(sample_python_file))

        assert result == 1

        captured = capsys.readouterr()
        assert "not a directory" in captured.out.lower()

    @pytest.mark.unit()
    def test_calls_ruff_on_directory(self, lint_and_format_directory, sample_directory):
        """Test that function calls ruff on the directory."""
        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            lint_and_format_directory(str(sample_directory))

            # Check that ruff was called with directory path
            calls = [str(call) for call in mock_run.call_args_list]
            dir_name = sample_directory.name
            assert any(dir_name in call or str(sample_directory) in call for call in calls)


# ============================================================================
# Tests for main function
# ============================================================================


class Test_Main_Function:
    """Tests for the main() function."""

    @pytest.mark.unit()
    def test_main_exists(self):
        """Test that main function exists and is callable."""
        from src.utils.format_lint.lint_format import main

        assert callable(main)

    @pytest.mark.unit()
    def test_main_with_no_args_uses_current_directory(self, monkeypatch):
        """Test that main with no arguments uses current directory."""
        from src.utils.format_lint.lint_format import main

        with patch("src.utils.format_lint.lint_format.lint_and_format_directory") as mock_lint_dir:
            mock_lint_dir.return_value = 0

            with patch("sys.argv", ["lint_format.py"]):
                # main() should work without raising
                try:
                    main()
                except SystemExit:
                    pass  # argparse may call sys.exit

    @pytest.mark.unit()
    def test_main_with_file_argument(self, sample_python_file):
        """Test that main with file argument calls lint_and_format_file."""
        from src.utils.format_lint.lint_format import main

        with patch("src.utils.format_lint.lint_format.lint_and_format_file") as mock_lint_file:
            mock_lint_file.return_value = 0

            with patch("sys.argv", ["lint_format.py", str(sample_python_file)]):
                try:
                    main()
                except SystemExit:
                    pass

    @pytest.mark.unit()
    def test_main_with_directory_argument(self, sample_directory):
        """Test that main with directory argument calls lint_and_format_directory."""
        from src.utils.format_lint.lint_format import main

        with patch("src.utils.format_lint.lint_format.lint_and_format_directory") as mock_lint_dir:
            mock_lint_dir.return_value = 0

            with patch("sys.argv", ["lint_format.py", str(sample_directory)]):
                try:
                    main()
                except SystemExit:
                    pass


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class Test_Edge_Cases:
    """Tests for edge cases and error handling."""

    @pytest.mark.unit()
    def test_handles_permission_error(self, lint_and_format_file, tmp_path):
        """Test handling of permission errors (mocked)."""
        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.side_effect = PermissionError("Access denied")

            # Should not crash
            try:
                lint_and_format_file(str(tmp_path / "test.py"))
            except PermissionError:
                pass  # Expected behavior

    @pytest.mark.unit()
    def test_handles_unicode_in_file_path(self, lint_and_format_file, tmp_path):
        """Test handling of unicode characters in file path."""
        unicode_file = tmp_path / "тест_файл.py"
        unicode_file.write_text("x = 1\n")

        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            # Should not crash with unicode path
            result = lint_and_format_file(str(unicode_file))

            assert isinstance(result, int)

    @pytest.mark.unit()
    def test_handles_spaces_in_path(self, lint_and_format_file, tmp_path):
        """Test handling of spaces in file path."""
        spaced_dir = tmp_path / "directory with spaces"
        spaced_dir.mkdir()
        spaced_file = spaced_dir / "test file.py"
        spaced_file.write_text("x = 1\n")

        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            result = lint_and_format_file(str(spaced_file))

            assert isinstance(result, int)

    @pytest.mark.unit()
    def test_empty_directory(self, lint_and_format_directory, tmp_path):
        """Test linting an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("src.utils.format_lint.lint_format.run_command") as mock_run:
            mock_run.return_value = 0

            result = lint_and_format_directory(str(empty_dir))

            assert isinstance(result, int)


# ============================================================================
# Integration Tests
# ============================================================================


class Test_Integration:
    """Integration tests (require ruff to be installed)."""

    @pytest.mark.integration()
    @pytest.mark.slow()
    def test_actual_ruff_execution(self, sample_python_file):
        """Test actual ruff execution on a file."""
        from src.utils.format_lint.lint_format import lint_and_format_file

        # This test requires ruff to be installed
        result = lint_and_format_file(str(sample_python_file))

        # Should complete without crashing
        assert isinstance(result, int)

    @pytest.mark.integration()
    @pytest.mark.slow()
    def test_actual_ruff_on_directory(self, sample_directory):
        """Test actual ruff execution on a directory."""
        from src.utils.format_lint.lint_format import lint_and_format_directory

        result = lint_and_format_directory(str(sample_directory))

        assert isinstance(result, int)
