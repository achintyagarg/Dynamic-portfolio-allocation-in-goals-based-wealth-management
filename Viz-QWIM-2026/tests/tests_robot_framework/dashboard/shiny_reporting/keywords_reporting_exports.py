"""
Robot Framework keyword library for reporting package public API tests.
======================================================================

Tests cover the ``src.dashboard.reporting`` package export surface after
the 2026-01 pyright cleanup that removed ``build_typst_data_context``
from ``reporting/__init__.py``.

Keywords exposed:
    - Reporting Package Should Be Importable
    - Symbol Should Be Exported        <symbol>
    - Symbol Should Be Callable        <symbol>
    - Symbol Should NOT Be Exported    <symbol>
    - All List Should Contain Exactly  <count>
    - All List Should Contain          <symbol>
    - All List Should Not Contain      <symbol>
    - Validate Polars DF With Valid DataFrame
    - Validate Polars DF With None Input

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-01
"""

from __future__ import annotations

import sys
import io
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path so src packages resolve correctly
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Robot Framework redirects sys.stderr to a StringIO object that lacks
# a .buffer attribute.  Patch it back before importing modules that
# access sys.stderr.buffer at module-load time.
# ---------------------------------------------------------------------------
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""
_reporting_pkg: Any = None

try:
    import src.dashboard.reporting as _reporting_pkg
    from src.dashboard.reporting import validate_polars_DF
    import polars as pl
except Exception as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)


def _require_imports() -> None:
    """Raise RuntimeError when reporting package could not be imported."""
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Reporting package could not be imported: {_import_error_message}"
        )


# ---------------------------------------------------------------------------
# Package availability
# ---------------------------------------------------------------------------


def reporting_package_should_be_importable() -> None:
    """Verify that ``src.dashboard.reporting`` is importable.

    Raises ``RuntimeError`` when the import failed, which Robot Framework
    reports as a FAIL with a descriptive message.
    """
    _require_imports()


# ---------------------------------------------------------------------------
# Export surface: positive checks
# ---------------------------------------------------------------------------


def symbol_should_be_exported(symbol: str) -> None:
    """Verify ``symbol`` exists in the ``src.dashboard.reporting`` namespace.

    Args:
        symbol: Name of the attribute to check.
    """
    _require_imports()
    if not hasattr(_reporting_pkg, symbol):
        raise AssertionError(
            f"'{symbol}' is not accessible from src.dashboard.reporting"
        )


def symbol_should_be_callable(symbol: str) -> None:
    """Verify ``symbol`` from the reporting package is callable.

    Args:
        symbol: Name of the attribute to check.
    """
    _require_imports()
    obj = getattr(_reporting_pkg, symbol, None)
    if not callable(obj):
        raise AssertionError(
            f"'{symbol}' from src.dashboard.reporting is not callable"
        )


# ---------------------------------------------------------------------------
# Negative guard
# ---------------------------------------------------------------------------


def symbol_should_not_be_exported(symbol: str) -> None:
    """Verify ``symbol`` does NOT exist in the reporting package namespace.

    This is the primary regression guard for ``build_typst_data_context``
    which was removed in the 2026-01 pyright cleanup.

    Args:
        symbol: Name that must not appear in the package namespace.
    """
    _require_imports()
    if hasattr(_reporting_pkg, symbol):
        raise AssertionError(
            f"'{symbol}' must NOT be exported from src.dashboard.reporting; "
            "the function was removed and its export was a bug"
        )


# ---------------------------------------------------------------------------
# __all__ checks
# ---------------------------------------------------------------------------


def all_list_should_contain_exactly(count: str) -> None:
    """Verify ``__all__`` has exactly ``count`` entries.

    Args:
        count: Expected number of entries (passed as string by Robot Framework).
    """
    _require_imports()
    expected: int = int(count)
    all_list: list[str] = getattr(_reporting_pkg, "__all__", [])
    if len(all_list) != expected:
        raise AssertionError(
            f"reporting.__all__ should have {expected} entries, "
            f"found {len(all_list)}: {all_list}"
        )


def all_list_should_contain(symbol: str) -> None:
    """Verify ``symbol`` appears in ``reporting.__all__``.

    Args:
        symbol: Symbol name that must be listed in ``__all__``.
    """
    _require_imports()
    all_list: list[str] = getattr(_reporting_pkg, "__all__", [])
    if symbol not in all_list:
        raise AssertionError(
            f"'{symbol}' not found in reporting.__all__: {all_list}"
        )


def all_list_should_not_contain(symbol: str) -> None:
    """Verify ``symbol`` does NOT appear in ``reporting.__all__``.

    Args:
        symbol: Symbol name that must not be listed in ``__all__``.
    """
    _require_imports()
    all_list: list[str] = getattr(_reporting_pkg, "__all__", [])
    if symbol in all_list:
        raise AssertionError(
            f"'{symbol}' unexpectedly found in reporting.__all__: {all_list}"
        )


# ---------------------------------------------------------------------------
# validate_polars_DF functional
# ---------------------------------------------------------------------------


def validate_polars_df_with_valid_dataframe() -> Any:
    """Call validate_polars_DF with a well-formed Polars DataFrame.

    Returns:
        The return value of ``validate_polars_DF``.

    Raises:
        AssertionError: if the function returns ``False``.
    """
    _require_imports()
    df = pl.DataFrame({"Date": ["2024-01-01", "2024-06-01"], "Value": [100.0, 110.0]})
    result = validate_polars_DF(df)
    if result is False:
        raise AssertionError(
            "validate_polars_DF returned False for a valid Polars DataFrame"
        )
    return result


def validate_polars_df_with_none_input() -> None:
    """Call validate_polars_DF with None; controlled exceptions are acceptable.

    Raises:
        AssertionError: if an unexpected (non-validation) exception propagates.
    """
    _require_imports()
    try:
        validate_polars_DF(None)  # type: ignore[arg-type]
    except (ValueError, TypeError, AttributeError):
        # Validation-level exceptions are expected and acceptable
        pass
    except Exception as exc:
        raise AssertionError(
            f"validate_polars_DF raised unexpected exception for None input: {exc}"
        ) from exc
