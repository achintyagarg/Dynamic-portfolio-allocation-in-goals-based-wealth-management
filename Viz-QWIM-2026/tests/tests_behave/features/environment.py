"""Behave environment hooks shared across all feature suites.

This module is executed by Behave before any scenario runs.  It:

* Ensures the project root is on ``sys.path`` so that ``src.*`` imports
  resolve correctly from every step definition file.
* Patches ``sys.stderr`` if Behave has replaced it with a ``StringIO``
  object that lacks a ``.buffer`` attribute (required by
  ``exception_custom.py``).
* Exposes ``context.project_root`` for use in step files that need to
  construct absolute paths (e.g. CSV data loaders).

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.stderr patch — must run before any src.* import that touches
# exception_custom.py (which calls sys.stderr.buffer at module level).
# ---------------------------------------------------------------------------
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Behave hooks
# ---------------------------------------------------------------------------


def before_all(context) -> None:
    """Store project root and reset shared state before the test run."""
    context.project_root = _PROJECT_ROOT


def before_scenario(context, scenario) -> None:
    """Reset per-scenario state so scenarios are fully independent."""
    context.portfolio = None
    context.etf_data = None
    context.portfolio_values = None
    context.benchmark_values = None
    context.client = None
    context.client2 = None
    context.simulation = None
    context.sim_results = None
    context.raw_data = None
    context.validation_result = None
    context.returns_series = None
    context.processed_df = None
    context.normalised_df = None
    context.transformed_df = None
    context.formatted_value = None
    context.extracted_value = None
    context.update_result = None
