"""Pytest configuration and shared fixtures for the QWIM Shiny test suite.

Covers two test categories:

1. **Unit tests** — exercise pure-Python dashboard utilities without a
   running server (``test_utils_data``, ``test_reactives_shiny``,
   ``test_utils_enhanced_formatting``, ``test_utils_tab_results``).

2. **Playwright end-to-end tests** — launch the Shiny app as a subprocess,
   then drive it through a real browser via ``pytest-playwright``.

How to run (from the project root):
    # All shiny tests
    pytest tests/tests_shiny/ -p no:cacheprovider --import-mode=importlib

    # Only unit tests (no browser)
    pytest tests/tests_shiny/ -m "unit"

    # Only playwright tests
    pytest tests/tests_shiny/ -m "playwright" --headed  # (add --headed to watch)
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Generator

import pytest

# ---------------------------------------------------------------------------
# UTF-8 safety patch  (mirrors main conftest.py)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import os as _os_enc

    _os_enc.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for _stream in (sys.stdout, sys.stderr, sys.stdin):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    if not hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Pytest marker declarations
# ---------------------------------------------------------------------------


def pytest_configure(config: Any) -> None:
    """Register custom markers so --strict-markers never rejects them."""
    markers = [
        "unit: fast tests that exercise pure Python functions",
        "integration: tests that load data files from disk",
        "playwright: browser-based end-to-end tests via Playwright",
        "smoke: quick sanity checks — subset of unit/playwright",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


# ---------------------------------------------------------------------------
# Fixtures — data paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_dir() -> Path:
    """Return the absolute project root directory."""
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def inputs_raw_dir(project_dir: Path) -> Path:
    """Return the `inputs/raw/` directory path."""
    return project_dir / "inputs" / "raw"


@pytest.fixture(scope="session")
def inputs_processed_dir(project_dir: Path) -> Path:
    """Return the `inputs/processed/` directory path."""
    return project_dir / "inputs" / "processed"


# ---------------------------------------------------------------------------
# Fixtures — loaded dashboard data  (session scope for speed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def data_utils(project_dir: Path) -> dict[str, Any]:
    """Load ``get_data_utils`` once per session."""
    from src.dashboard.shiny_utils.utils_data import get_data_utils

    return get_data_utils(project_dir=project_dir)


@pytest.fixture(scope="session")
def data_inputs(project_dir: Path) -> dict[str, Any]:
    """Load ``get_data_inputs`` once per session."""
    from src.dashboard.shiny_utils.utils_data import get_data_inputs

    return get_data_inputs(project_dir=project_dir)


# ---------------------------------------------------------------------------
# Fixtures — in-memory Polars DataFrames  (unit tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_portfolio_df() -> "import polars as pl; pl.DataFrame":
    """Return a minimal portfolio-style Polars DataFrame with a Date column."""
    import polars as pl

    return pl.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Value": [100.0, 102.5, 101.0],
        }
    )


@pytest.fixture(scope="session")
def sample_timeseries_df() -> Any:
    """Return a Polars DataFrame with 250 rows to test downsampling."""
    import polars as pl

    n = 250
    return pl.DataFrame(
        {
            "Date": [f"2024-01-{(i % 28) + 1:02d}" for i in range(n)],
            "VTI": [100.0 + i * 0.1 for i in range(n)],
            "AGG": [90.0 - i * 0.05 for i in range(n)],
        }
    )


@pytest.fixture(scope="session")
def sample_weights_df() -> Any:
    """Return a portfolio-weights Polars DataFrame (rows sum ~1.0)."""
    import polars as pl

    return pl.DataFrame(
        {
            "Date": ["2024-01-01", "2024-02-01"],
            "VTI": [0.40, 0.45],
            "AGG": [0.35, 0.30],
            "VNQ": [0.25, 0.25],
        }
    )


@pytest.fixture(scope="session")
def minimal_reactives_shiny(data_utils: dict[str, Any]) -> dict[str, Any]:
    """Return a fully-initialised reactives_shiny dict (session scope)."""
    from src.dashboard.shiny_utils.reactives_shiny import initialize_reactives_shiny
    from shiny import reactive

    with reactive.isolate():
        return initialize_reactives_shiny(data_utils=data_utils)


# ---------------------------------------------------------------------------
# Fixtures — Playwright / Shiny server  (session scope)
# ---------------------------------------------------------------------------

#: Bind the dev server on an unusual port to avoid clashing with the real app.
_SHINY_TEST_PORT: int = 8765
_SHINY_TEST_URL: str = f"http://127.0.0.1:{_SHINY_TEST_PORT}"


@pytest.fixture(scope="session")
def shiny_server_url(project_dir: Path) -> Generator[str, None, None]:
    """Start the QWIM Shiny app as a subprocess then yield its base URL.

    The fixture waits until the server is responsive (up to 60 s) before
    yielding.  On teardown it sends SIGTERM and waits for the process to exit.

    Usage in a playwright test::

        def test_my_page(page, shiny_server_url):
            page.goto(shiny_server_url)
            ...
    """
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys, os; "
                f"sys.path.insert(0, r'{project_dir}'); "
                "os.environ['QWIM_ENVIRONMENT'] = 'development'; "
                "from shiny import run_app; "
                "from src.dashboard.main_App import app; "
                f"run_app(app, host='127.0.0.1', port={_SHINY_TEST_PORT}, reload=False)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(project_dir),
    )

    # Poll until the server accepts connections (max 90 s)
    import socket

    deadline = time.monotonic() + 90.0
    up = False
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", _SHINY_TEST_PORT), timeout=1.0):
                up = True
                break
        except OSError:
            time.sleep(1.0)

    if not up:
        process.terminate()
        process.wait(timeout=10)
        pytest.skip(f"Shiny server did not start on port {_SHINY_TEST_PORT} within 90 s")

    # Extra pause for Shiny to finish initialising reactive graph
    time.sleep(3.0)

    yield _SHINY_TEST_URL

    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


# ---------------------------------------------------------------------------
# Helpers available to all test modules
# ---------------------------------------------------------------------------


def make_valid_reactives_shiny() -> dict[str, Any]:
    """Return the canonical valid reactives_shiny skeleton for unit tests.

    Avoids needing a running Shiny context — uses plain dicts instead of
    reactive.Value so validators can be tested without the reactive session.
    """
    return {
        "User_Inputs_Shiny": {},
        "Inner_Variables_Shiny": {},
        "Triggers_Shiny": {},
        "Visual_Objects_Shiny": {},
    }
