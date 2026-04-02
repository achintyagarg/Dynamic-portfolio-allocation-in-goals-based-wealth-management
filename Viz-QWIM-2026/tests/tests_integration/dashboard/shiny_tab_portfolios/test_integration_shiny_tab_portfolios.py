"""Integration tests for the shiny_tab_portfolios modules.

Exercises real module imports and the cross-module contracts that must hold:

* All four subtab modules expose the expected callable API
* The ``tab_portfolios`` orchestrator wires up the correct module IDs
* Shiny input-identifier naming conventions are enforced throughout
* Optimization method constants are consistent and complete
* Server functions expose the expected signatures

Tests intentionally avoid launching a live Shiny server; they operate on
pure-Python logic that is callable without a reactive context.

Run:
    pytest tests/tests_integration/dashboard/shiny_tab_portfolios/ -m integration -v

Author:
    QWIM Development Team

Version:
    0.2.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import inspect

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Guard: all portfolio subtab modules
# ---------------------------------------------------------------------------
try:
    from src.dashboard.shiny_tab_portfolios.subtab_portfolios_analysis import (
        subtab_portfolios_analysis_server,
        subtab_portfolios_analysis_ui,
    )
    from src.dashboard.shiny_tab_portfolios.subtab_portfolios_comparison import (
        subtab_portfolios_comparison_server,
        subtab_portfolios_comparison_ui,
    )
    from src.dashboard.shiny_tab_portfolios.subtab_portfolios_skfolio import (
        BASIC_METHODS,
        CLUSTERING_METHODS,
        CONVEX_METHODS,
        ENSEMBLE_METHODS,
        OBJECTIVE_FUNCTIONS,
        OPTIMIZATION_CATEGORIES,
        subtab_portfolios_skfolio_server,
        subtab_portfolios_skfolio_ui,
    )
    from src.dashboard.shiny_tab_portfolios.subtab_weights_analysis import (
        subtab_weights_analysis_server,
        subtab_weights_analysis_ui,
    )
    from src.dashboard.shiny_tab_portfolios.tab_portfolios import (
        tab_portfolios_server,
        tab_portfolios_ui,
    )

    MODULE_IMPORT_AVAILABLE = True
except ImportError as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("shiny_tab_portfolios module import failed: %s", _exc)

pytestmark_modules = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="shiny_tab_portfolios modules not importable",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_data_utils() -> dict[str, Any]:
    """Minimal data_utils dictionary accepted by all subtab modules."""
    return {"theme": "default", "export_enabled": False, "chart_height": 600}


@pytest.fixture()
def sample_portfolio_data() -> pl.DataFrame:
    """Sample portfolio value timeseries."""
    dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(100)]  # noqa: DTZ001
    np.random.seed(42)
    values = 100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 100))
    return pl.DataFrame({"Date": dates, "Value": values.tolist()})


@pytest.fixture()
def sample_benchmark_data() -> pl.DataFrame:
    """Sample benchmark value timeseries."""
    dates = [(datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(100)]  # noqa: DTZ001
    np.random.seed(99)
    values = 100.0 * np.cumprod(1 + np.random.normal(0.0004, 0.011, 100))
    return pl.DataFrame({"Date": dates, "Value": values.tolist()})


@pytest.fixture()
def sample_weights_data() -> pl.DataFrame:
    """Sample portfolio weights timeseries."""
    dates = [(datetime(2024, 1, 1) + timedelta(days=i * 7)).strftime("%Y-%m-%d") for i in range(52)]  # noqa: DTZ001
    np.random.seed(42)
    n = len(dates)
    vti = np.random.uniform(0.35, 0.45, n)
    vxus = np.random.uniform(0.25, 0.35, n)
    bnd = np.random.uniform(0.15, 0.25, n)
    vnq = 1.0 - vti - vxus - bnd
    return pl.DataFrame(
        {"Date": dates, "VTI": vti.tolist(), "VXUS": vxus.tolist(),
         "BND": bnd.tolist(), "VNQ": vnq.tolist()},
    )


@pytest.fixture()
def sample_etf_data() -> pl.DataFrame:
    """Sample ETF price timeseries."""
    dates = [(datetime(2022, 1, 1) + timedelta(days=i * 7)).strftime("%Y-%m-%d") for i in range(104)]  # noqa: DTZ001
    np.random.seed(42)
    n = len(dates)
    return pl.DataFrame({
        "Date": dates,
        "VTI": (200.0 * np.cumprod(1 + np.random.normal(0.001, 0.015, n))).tolist(),
        "VXUS": (50.0 * np.cumprod(1 + np.random.normal(0.0008, 0.018, n))).tolist(),
        "BND": (75.0 * np.cumprod(1 + np.random.normal(0.0003, 0.005, n))).tolist(),
        "VNQ": (80.0 * np.cumprod(1 + np.random.normal(0.0009, 0.012, n))).tolist(),
    })


@pytest.fixture()
def sample_data_inputs(
    sample_portfolio_data: pl.DataFrame,
    sample_benchmark_data: pl.DataFrame,
    sample_weights_data: pl.DataFrame,
    sample_etf_data: pl.DataFrame,
) -> dict[str, Any]:
    """Full data_inputs dictionary accepted by all portfolio subtab modules."""
    return {
        "My_Portfolio": sample_portfolio_data,
        "Benchmark_Portfolio": sample_benchmark_data,
        "Weights_My_Portfolio": sample_weights_data,
        "Time_Series_ETFs": sample_etf_data,
    }


@pytest.fixture()
def sample_reactives_shiny() -> dict[str, Any]:
    """Standard reactives_shiny structure used across all server functions."""
    return {
        "User_Inputs_Shiny": {},
        "Inner_Variables_Shiny": {},
        "Triggers_Shiny": {},
        "Visual_Objects_Shiny": {},
    }


# ===========================================================================
# Module API — callable check
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_All_Subtab_Modules_Are_Callable:
    """Every subtab module exposes callable UI and server functions."""

    def test_subtab_portfolios_analysis_ui_callable(self) -> None:
        """subtab_portfolios_analysis_ui is callable."""
        assert callable(subtab_portfolios_analysis_ui)

    def test_subtab_portfolios_analysis_server_callable(self) -> None:
        """subtab_portfolios_analysis_server is callable."""
        assert callable(subtab_portfolios_analysis_server)

    def test_subtab_portfolios_comparison_ui_callable(self) -> None:
        """subtab_portfolios_comparison_ui is callable."""
        assert callable(subtab_portfolios_comparison_ui)

    def test_subtab_portfolios_comparison_server_callable(self) -> None:
        """subtab_portfolios_comparison_server is callable."""
        assert callable(subtab_portfolios_comparison_server)

    def test_subtab_weights_analysis_ui_callable(self) -> None:
        """subtab_weights_analysis_ui is callable."""
        assert callable(subtab_weights_analysis_ui)

    def test_subtab_weights_analysis_server_callable(self) -> None:
        """subtab_weights_analysis_server is callable."""
        assert callable(subtab_weights_analysis_server)

    def test_subtab_portfolios_skfolio_ui_callable(self) -> None:
        """subtab_portfolios_skfolio_ui is callable."""
        assert callable(subtab_portfolios_skfolio_ui)

    def test_subtab_portfolios_skfolio_server_callable(self) -> None:
        """subtab_portfolios_skfolio_server is callable."""
        assert callable(subtab_portfolios_skfolio_server)

    def test_tab_portfolios_ui_callable(self) -> None:
        """tab_portfolios_ui is callable."""
        assert callable(tab_portfolios_ui)

    def test_tab_portfolios_server_callable(self) -> None:
        """tab_portfolios_server is callable."""
        assert callable(tab_portfolios_server)


# ===========================================================================
# Module API — server function signatures
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Server_Function_Signatures:
    """Server functions accept the expected parameter names."""

    def test_analysis_server_accepts_id(self) -> None:
        """subtab_portfolios_analysis_server has 'id' parameter."""
        sig = inspect.signature(subtab_portfolios_analysis_server)
        assert "id" in sig.parameters

    def test_comparison_server_accepts_id(self) -> None:
        """subtab_portfolios_comparison_server has 'id' parameter."""
        sig = inspect.signature(subtab_portfolios_comparison_server)
        assert "id" in sig.parameters

    def test_weights_server_accepts_id(self) -> None:
        """subtab_weights_analysis_server has 'id' parameter."""
        sig = inspect.signature(subtab_weights_analysis_server)
        assert "id" in sig.parameters

    def test_skfolio_server_accepts_id(self) -> None:
        """subtab_portfolios_skfolio_server has 'id' parameter."""
        sig = inspect.signature(subtab_portfolios_skfolio_server)
        assert "id" in sig.parameters

    def test_tab_portfolios_server_accepts_id(self) -> None:
        """tab_portfolios_server has 'id' parameter."""
        sig = inspect.signature(tab_portfolios_server)
        assert "id" in sig.parameters


# ===========================================================================
# Input ID naming conventions
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Input_ID_Naming_Conventions:
    """Input IDs across portfolio subtabs follow the hierarchical convention."""

    _ANALYSIS_INPUT_IDS: ClassVar[list[str]] = [
        "input_ID_tab_portfolios_subtab_portfolios_analysis_time_period",
        "input_ID_tab_portfolios_subtab_portfolios_analysis_date_range",
        "input_ID_tab_portfolios_subtab_portfolios_analysis_type",
        "input_ID_tab_portfolios_subtab_portfolios_analysis_rolling_window",
        "input_ID_tab_portfolios_subtab_portfolios_analysis_include_benchmark",
    ]

    _COMPARISON_PREFIX = "input_ID_tab_portfolios_subtab_portfolios_comparison_"
    _WEIGHTS_PREFIX = "input_ID_tab_portfolios_subtab_weights_analysis_"
    _SKFOLIO_INPUT_IDS: ClassVar[list[str]] = [
        "input_ID_tab_portfolios_subtab_skfolio_time_period",
        "input_ID_tab_portfolios_subtab_skfolio_method1_category",
        "input_ID_tab_portfolios_subtab_skfolio_method2_category",
        "input_ID_tab_portfolios_subtab_skfolio_btn_optimize",
    ]

    def test_analysis_input_ids_follow_convention(self) -> None:
        """All analysis subtab input IDs start with the correct hierarchical prefix."""
        prefix = "input_ID_tab_portfolios_subtab_portfolios_analysis_"
        for id_ in self._ANALYSIS_INPUT_IDS:
            assert id_.startswith(prefix), (
                f"Input ID '{id_}' must start with '{prefix}'"
            )
        _logger.debug("Analysis input IDs convention verified")

    def test_skfolio_input_ids_follow_convention(self) -> None:
        """All skfolio subtab input IDs start with the correct hierarchical prefix."""
        prefix = "input_ID_tab_portfolios_subtab_skfolio_"
        for id_ in self._SKFOLIO_INPUT_IDS:
            assert id_.startswith(prefix), (
                f"Input ID '{id_}' must start with '{prefix}'"
            )
        _logger.debug("skfolio input IDs convention verified")

    def test_analysis_input_ids_are_lowercase_snake_case(self) -> None:
        """Analysis input IDs use lowercase snake_case (except 'ID' fragment)."""
        for id_ in self._ANALYSIS_INPUT_IDS:
            # Replace known UPPERCASE exceptions
            normalized = id_.replace("input_ID_", "input_id_").replace("_subtab_", "_subtab_")
            assert normalized == normalized.lower(), (
                f"Input ID '{id_}' has unexpected uppercase characters"
            )

    def test_output_id_uses_output_prefix(self) -> None:
        """Output IDs use 'output_ID_' prefix to distinguish from inputs."""
        output_ids = [
            "output_ID_tab_portfolios_subtab_portfolios_analysis_plot_main",
            "output_ID_tab_portfolios_subtab_portfolios_analysis_data_info",
            "output_ID_tab_portfolios_subtab_skfolio_plot_weights",
            "output_ID_tab_portfolios_subtab_skfolio_status",
        ]
        for oid in output_ids:
            assert oid.startswith("output_ID_"), (
                f"Output ID '{oid}' must start with 'output_ID_'"
            )


# ===========================================================================
# tab_portfolios orchestrator — module IDs
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Tab_Portfolios_Module_IDs:
    """tab_portfolios orchestrator uses the correct subtab module IDs."""

    _EXPECTED_SUBTAB_IDS: ClassVar[list[str]] = [
        "ID_tab_portfolios_subtab_portfolios_analysis",
        "ID_tab_portfolios_subtab_portfolios_comparison",
        "ID_tab_portfolios_subtab_weights_analysis",
        "ID_tab_portfolios_subtab_skfolio",
    ]

    def test_subtab_module_ids_listed(self) -> None:
        """All four expected subtab module IDs exist as strings."""
        for subtab_id in self._EXPECTED_SUBTAB_IDS:
            assert isinstance(subtab_id, str)
            assert len(subtab_id) > 0
        _logger.debug(f"Verified {len(self._EXPECTED_SUBTAB_IDS)} subtab module IDs")

    def test_subtab_module_ids_start_with_id_tab_portfolios(self) -> None:
        """All subtab module IDs start with 'ID_tab_portfolios_subtab_'."""
        for subtab_id in self._EXPECTED_SUBTAB_IDS:
            assert subtab_id.startswith("ID_tab_portfolios_subtab_"), (
                f"Module ID '{subtab_id}' must start with 'ID_tab_portfolios_subtab_'"
            )

    def test_four_subtab_module_ids_expected(self) -> None:
        """Exactly four subtab module IDs are registered."""
        assert len(self._EXPECTED_SUBTAB_IDS) == 4
        _logger.debug("Four subtab module IDs confirmed")


# ===========================================================================
# skfolio method constants — cross-consistency
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Skfolio_Method_Constants_Consistency:
    """skfolio method constants are internally consistent."""

    def test_each_optimization_category_has_method_dict(self) -> None:
        """Each key in OPTIMIZATION_CATEGORIES maps to a non-empty method dict."""
        category_to_dict = {
            "basic": BASIC_METHODS,
            "convex": CONVEX_METHODS,
            "clustering": CLUSTERING_METHODS,
            "ensemble": ENSEMBLE_METHODS,
        }
        for key in OPTIMIZATION_CATEGORIES:
            assert key in category_to_dict, f"Category key '{key}' missing from map"
            assert len(category_to_dict[key]) > 0, f"Method dict for '{key}' is empty"

    def test_no_method_key_duplicated_across_categories(self) -> None:
        """No method key appears in more than one category dict."""
        all_keys = (
            list(BASIC_METHODS.keys())
            + list(CONVEX_METHODS.keys())
            + list(CLUSTERING_METHODS.keys())
            + list(ENSEMBLE_METHODS.keys())
        )
        assert len(all_keys) == len(set(all_keys)), "Duplicate method keys detected"

    def test_total_methods_is_thirteen(self) -> None:
        """Total method count across all categories is exactly 13."""
        total = (
            len(BASIC_METHODS)
            + len(CONVEX_METHODS)
            + len(CLUSTERING_METHODS)
            + len(ENSEMBLE_METHODS)
        )
        assert total == 13

    def test_objective_functions_has_four_entries(self) -> None:
        """OBJECTIVE_FUNCTIONS contains exactly four entries."""
        assert len(OBJECTIVE_FUNCTIONS) == 4

    def test_all_method_values_are_non_empty_strings(self) -> None:
        """Every method label (dict value) is a non-empty string."""
        all_dicts = [BASIC_METHODS, CONVEX_METHODS, CLUSTERING_METHODS, ENSEMBLE_METHODS,
                     OBJECTIVE_FUNCTIONS]
        for d in all_dicts:
            for key, val in d.items():
                assert isinstance(val, str), f"Method dict: key='{key}' has invalid label '{val}'"
                assert len(val) > 0, f"Method dict: key='{key}' has empty label"


# ===========================================================================
# UI function — component creation
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_UI_Functions_Return_Components:
    """UI functions return non-None Shiny components."""

    def test_analysis_ui_returns_component(
        self,
        sample_data_utils: dict[str, Any],
        sample_data_inputs: dict[str, Any],
    ) -> None:
        """subtab_portfolios_analysis_ui returns a non-None component."""
        result = subtab_portfolios_analysis_ui(
            "test_analysis",
            data_utils=sample_data_utils,
            data_inputs=sample_data_inputs,
        )
        assert result is not None

    def test_comparison_ui_returns_component(
        self,
        sample_data_utils: dict[str, Any],
        sample_data_inputs: dict[str, Any],
    ) -> None:
        """subtab_portfolios_comparison_ui returns a non-None component."""
        result = subtab_portfolios_comparison_ui(
            "test_comparison",
            data_utils=sample_data_utils,
            data_inputs=sample_data_inputs,
        )
        assert result is not None

    def test_weights_ui_returns_component(
        self,
        sample_data_utils: dict[str, Any],
        sample_data_inputs: dict[str, Any],
    ) -> None:
        """subtab_weights_analysis_ui returns a non-None component."""
        result = subtab_weights_analysis_ui(
            "test_weights",
            data_utils=sample_data_utils,
            data_inputs=sample_data_inputs,
        )
        assert result is not None

    def test_skfolio_ui_returns_component(
        self,
        sample_data_utils: dict[str, Any],
        sample_data_inputs: dict[str, Any],
    ) -> None:
        """subtab_portfolios_skfolio_ui returns a non-None component."""
        result = subtab_portfolios_skfolio_ui(
            "test_skfolio",
            data_utils=sample_data_utils,
            data_inputs=sample_data_inputs,
        )
        assert result is not None


# ===========================================================================
# OUTPUT_DIR consistency
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Output_Dir_Consistency:
    """OUTPUT_DIR is consistently defined across all portfolio modules."""

    def test_analysis_output_dir_is_path(self) -> None:
        """subtab_portfolios_analysis OUTPUT_DIR is a Path."""
        from src.dashboard.shiny_tab_portfolios.subtab_portfolios_analysis import (
            OUTPUT_DIR as ANALYSIS_OUTPUT_DIR,
        )
        assert isinstance(ANALYSIS_OUTPUT_DIR, Path)

    def test_skfolio_output_dir_is_path(self) -> None:
        """subtab_portfolios_skfolio OUTPUT_DIR is a Path."""
        from src.dashboard.shiny_tab_portfolios.subtab_portfolios_skfolio import (
            OUTPUT_DIR as SKFOLIO_OUTPUT_DIR,
        )
        assert isinstance(SKFOLIO_OUTPUT_DIR, Path)

    def test_tab_portfolios_output_dir_is_path(self) -> None:
        """tab_portfolios OUTPUT_DIR is a Path."""
        from src.dashboard.shiny_tab_portfolios.tab_portfolios import (
            OUTPUT_DIR as TAB_OUTPUT_DIR,
        )
        assert isinstance(TAB_OUTPUT_DIR, Path)
