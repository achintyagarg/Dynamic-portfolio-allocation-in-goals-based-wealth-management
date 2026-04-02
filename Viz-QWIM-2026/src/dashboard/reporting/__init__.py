"""Reporting module for the QWIM Dashboard.

Provides Typst-based PDF report generation functionality.
"""

from __future__ import annotations

from .report_QWIM import (
    compile_typst_report,
    generate_report_PDF,
    validate_polars_DF,
)


__all__ = [
    "compile_typst_report",
    "generate_report_PDF",
    "validate_polars_DF",
]
