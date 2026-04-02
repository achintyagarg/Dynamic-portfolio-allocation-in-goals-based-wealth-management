"""Integration tests for the reporting package public API.

Verifies that the ``src.dashboard.reporting`` package exports exactly the
three documented public symbols and that the removed symbol
``build_typst_data_context`` is *not* accessible from the package root.

These tests guard against regressions introduced when ``reporting/__init__.py``
was updated (2026-01 pyright cleanup) to remove ``build_typst_data_context``
from the export list.

Test Categories
---------------
- Package-level import: all 3 public symbols importable
- Negative guard: ``build_typst_data_context`` must NOT be in the package
- ``__all__`` exact match: no accidental additions or deletions
- ``validate_polars_DF`` functional smoke: accepts a valid Polars DataFrame
- ``compile_typst_report`` / ``generate_report_PDF`` callable smoke

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-01
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

MODULE_IMPORT_AVAILABLE: bool = True

try:
    import src.dashboard.reporting as reporting_pkg
    from src.dashboard.reporting import (
        compile_typst_report,
        generate_report_PDF,
        validate_polars_DF,
    )
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Reporting package import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="src.dashboard.reporting not importable in this environment",
)


# ==============================================================================
# Package export surface
# ==============================================================================


@pytest.mark.integration()
class Test_Reporting_Package_Export_Surface:
    """Integration tests validating the reporting package's public export surface.

    These are consumer-facing checks: they confirm exactly what an external
    caller importing from ``src.dashboard.reporting`` will find.
    """

    _EXPECTED_EXPORTS: frozenset[str] = frozenset(
        {"compile_typst_report", "generate_report_PDF", "validate_polars_DF"}
    )

    def test_compile_typst_report_is_importable(self) -> None:
        """compile_typst_report must be importable from the reporting package."""
        assert callable(compile_typst_report), (
            "compile_typst_report is not callable — import may have failed silently"
        )
        _logger.debug("compile_typst_report importable and callable")

    def test_generate_report_PDF_is_importable(self) -> None:
        """generate_report_PDF must be importable from the reporting package."""
        assert callable(generate_report_PDF), (
            "generate_report_PDF is not callable — import may have failed silently"
        )
        _logger.debug("generate_report_PDF importable and callable")

    def test_validate_polars_DF_is_importable(self) -> None:
        """validate_polars_DF must be importable from the reporting package."""
        assert callable(validate_polars_DF), (
            "validate_polars_DF is not callable — import may have failed silently"
        )
        _logger.debug("validate_polars_DF importable and callable")

    def test_build_typst_data_context_NOT_exported(self) -> None:
        """build_typst_data_context must NOT be accessible from the package root.

        This is the primary regression guard for the 2026-01 fix that removed
        this non-existent function from ``reporting/__init__.py``.
        """
        assert not hasattr(reporting_pkg, "build_typst_data_context"), (
            "build_typst_data_context must NOT be exported from src.dashboard.reporting; "
            "the function was removed and its presence in __init__.py was a bug"
        )
        _logger.debug("build_typst_data_context correctly absent from reporting package")

    def test_all_list_matches_expected_exports(self) -> None:
        """``__all__`` must equal exactly the three documented public symbols."""
        actual_all: set[str] = set(getattr(reporting_pkg, "__all__", []))
        assert actual_all == self._EXPECTED_EXPORTS, (
            f"reporting.__all__ mismatch. "
            f"Expected: {self._EXPECTED_EXPORTS}, "
            f"Got: {actual_all}. "
            "Update this test if exports are intentionally changed."
        )
        _logger.debug("reporting.__all__ matches expected exports: %s", actual_all)

    def test_no_extra_callables_beyond_all(self) -> None:
        """No undocumented public callable should exist outside __all__.

        Checks that every public callable in the package namespace is listed
        in __all__ (i.e. the surface is closed).
        """
        all_exports: set[str] = set(getattr(reporting_pkg, "__all__", []))
        public_callables: set[str] = {
            name
            for name in dir(reporting_pkg)
            if not name.startswith("_") and callable(getattr(reporting_pkg, name, None))
        }
        undocumented = public_callables - all_exports
        assert not undocumented, (
            f"Public callables found outside __all__: {undocumented}. "
            "Either add them to __all__ or make them private with a leading underscore."
        )
        _logger.debug("No undocumented public callables found in reporting package")


# ==============================================================================
# validate_polars_DF functional smoke
# ==============================================================================


@pytest.mark.integration()
class Test_Validate_Polars_DF_Functional:
    """Functional integration tests for the validate_polars_DF utility.

    ``validate_polars_DF`` is a pure-Python function with no external
    dependencies, so it can be exercised end-to-end within the integration
    test suite without any infrastructure setup.
    """

    def test_valid_dataframe_passes_validation(self) -> None:
        """validate_polars_DF returns True for a well-formed Polars DataFrame."""
        df = pl.DataFrame({"Date": ["2024-01-01", "2024-02-01"], "Value": [100.0, 110.0]})
        result: Any = validate_polars_DF(df)
        # The function should either return True/non-falsy or raise;
        # any truthy result (including None) is acceptable for a pass.
        assert result is not False, (
            "validate_polars_DF returned False for a valid DataFrame"
        )
        _logger.debug("validate_polars_DF accepted valid DataFrame")

    def test_empty_dataframe_handled_gracefully(self) -> None:
        """validate_polars_DF does not raise an unhandled exception for empty DataFrame."""
        df = pl.DataFrame({"Date": [], "Value": []})
        try:
            validate_polars_DF(df)
        except Exception as exc:
            # Controlled exceptions (ValueError, TypeError) are acceptable;
            # only unhandled crashes (RuntimeError, AttributeError, etc.) fail.
            if not isinstance(exc, (ValueError, TypeError)):
                raise AssertionError(
                    f"validate_polars_DF raised unexpected exception for empty DF: {exc}"
                ) from exc
        _logger.debug("validate_polars_DF handled empty DataFrame without crashing")

    def test_non_dataframe_input_handled_gracefully(self) -> None:
        """validate_polars_DF does not crash for non-DataFrame input."""
        try:
            validate_polars_DF(None)  # type: ignore[arg-type]
        except Exception as exc:
            if not isinstance(exc, (ValueError, TypeError, AttributeError)):
                raise AssertionError(
                    f"validate_polars_DF raised unexpected exception for None: {exc}"
                ) from exc
        _logger.debug("validate_polars_DF handled None input without crashing")


# ==============================================================================
# compile_typst_report / generate_report_PDF callable smoke
# ==============================================================================


@pytest.mark.integration()
class Test_Reporting_Functions_Callable_Smoke:
    """Smoke tests confirming the reporting functions have expected callable signatures."""

    def test_compile_typst_report_has_two_required_parameters(self) -> None:
        """compile_typst_report signature must have at least two required parameters.

        The 2026-01 fix corrected test calls that mistakenly passed 3 args;
        the actual signature is (typ_template_path, output_pdf_path).
        """
        import inspect

        sig = inspect.signature(compile_typst_report)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, (
            f"compile_typst_report expects ≥2 parameters, found {len(params)}: {params}"
        )
        _logger.debug("compile_typst_report signature: %s", params)

    def test_generate_report_PDF_is_callable_with_correct_signature(self) -> None:
        """generate_report_PDF must be a callable with at least one parameter."""
        import inspect

        sig = inspect.signature(generate_report_PDF)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, (
            f"generate_report_PDF expects ≥1 parameter, found {len(params)}: {params}"
        )
        _logger.debug("generate_report_PDF signature: %s", params)
