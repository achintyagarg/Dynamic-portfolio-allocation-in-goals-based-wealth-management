"""
Linting and formatting utility for Python files using Ruff.

This module provides utilities for running Ruff linting and formatting
on Python files in the project.

Functions
---------
    run_command: Execute a shell command and capture output.
    lint_and_format_file: Run Ruff check and format on a specified file.
    lint_and_format_directory: Run Ruff check and format on a directory.
    main: Entry point for command-line usage.
"""

import argparse
import subprocess
import sys

from pathlib import Path


def run_command(
    command: str,
) -> int:
    """Run a shell command and print its output.

    Parameters
    ----------
    command : str
        The shell command to execute.

    Returns
    -------
    int
        The return code from the command execution.
    """
    result = subprocess.run(  # noqa: S602
        command,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode


def lint_and_format_file(
    file_path: str,
) -> int:
    """Run Ruff linting and formatting on a single file.

    Parameters
    ----------
    file_path : str
        Path to the Python file to lint and format.

    Returns
    -------
    int
        Final return code (0 if all checks pass).

    Notes
    -----
    This function performs the following steps:
    1. Check for issues using ruff check
    2. Auto-fix issues that can be fixed automatically
    3. Format the file using ruff format
    4. Run a final check for remaining issues
    """
    # Validate file path
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return 1

    if path.suffix != ".py":
        print(f"Warning: {file_path} is not a Python file.")

    print(f"Step 1: Checking for issues in {file_path}")
    run_command(f"ruff check {file_path}")

    print(f"\nStep 2: Fixing automatically fixable issues in {file_path}")
    run_command(f"ruff check --fix {file_path}")

    print(f"\nStep 3: Formatting {file_path}")
    run_command(f"ruff format {file_path}")

    print(f"\nStep 4: Final check for remaining issues in {file_path}")
    final_code = run_command(f"ruff check {file_path}")

    print("\nLinting and formatting completed!")
    return final_code


def lint_and_format_directory(
    directory_path: str,
    recursive: bool = True,  # noqa: ARG001
) -> int:
    """Run Ruff linting and formatting on a directory.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing Python files.
    recursive : bool, optional
        Whether to process subdirectories recursively. Default is True.

    Returns
    -------
    int
        Final return code (0 if all checks pass).
    """
    # Validate directory path
    path = Path(directory_path)
    if not path.exists():
        print(f"Error: Directory not found: {directory_path}")
        return 1

    if not path.is_dir():
        print(f"Error: {directory_path} is not a directory.")
        return 1

    print(f"Step 1: Checking for issues in {directory_path}")
    run_command(f"ruff check {directory_path}")

    print(f"\nStep 2: Fixing automatically fixable issues in {directory_path}")
    run_command(f"ruff check --fix {directory_path}")

    print(f"\nStep 3: Formatting {directory_path}")
    run_command(f"ruff format {directory_path}")

    print(f"\nStep 4: Final check for remaining issues in {directory_path}")
    final_code = run_command(f"ruff check {directory_path}")

    print("\nLinting and formatting completed!")
    return final_code


def main(
    target_path: str | None = None,
) -> None:
    """Entry point for linting and formatting utility.

    Parameters
    ----------
    target_path : str, optional
        Path to file or directory to lint. If None, uses command-line arguments
        or defaults to the current directory.

    Notes
    -----
    When run from command line, accepts the following arguments:
        - path: Path to file or directory (optional, defaults to current directory)
        - --file, -f: Treat path as a single file
        - --directory, -d: Treat path as a directory (default)

    Examples
    --------
    Command line usage:

    .. code-block:: bash

        # Lint entire src directory
        python lint_format.py src/

        # Lint a single file
        python lint_format.py -f src/utils/utils_portfolio.py

        # Lint current directory
        python lint_format.py
    """
    if target_path is not None:
        # Called programmatically
        path = Path(target_path)
        if path.is_file():
            lint_and_format_file(str(path))
        else:
            lint_and_format_directory(str(path))
        return

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Ruff linting and formatting on Python files.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to file or directory (default: current directory)",
    )
    parser.add_argument(
        "--file",
        "-f",
        action="store_true",
        help="Treat path as a single file",
    )
    parser.add_argument(
        "--directory",
        "-d",
        action="store_true",
        help="Treat path as a directory (default behavior)",
    )

    args = parser.parse_args()

    path = Path(args.path)

    if args.file or path.is_file():
        lint_and_format_file(str(path))
    else:
        lint_and_format_directory(str(path))


if __name__ == "__main__":
    main()
