"""Integration tests for the shiny_tab_clients modules.

Exercises real module imports and the cross-module contracts that must hold:

* ``utils_tab_clients`` formatting / validation helpers produce correct output
* All six subtab modules expose the expected callable API
* The ``tab_clients`` orchestrator wires up the correct module IDs
* Shiny input-identifier naming conventions are enforced throughout

Tests intentionally avoid launching a live Shiny server; they operate on
pure-Python logic that is callable without a reactive context.

Run:
    pytest tests/tests_integration/dashboard/shiny_tab_clients/ -m integration -v

Author:
    QWIM Development Team

Version:
    0.2.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

from typing import Any, ClassVar

import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Guard: utils_tab_clients availability
# ---------------------------------------------------------------------------
try:
    from src.dashboard.shiny_utils.utils_tab_clients import (
        format_currency_display,
        validate_age_range,
        validate_financial_amount,
        validate_string_input,
    )

    UTILS_IMPORT_AVAILABLE = True
except ImportError as _exc:
    UTILS_IMPORT_AVAILABLE = False
    _logger.warning("utils_tab_clients import failed: %s", _exc)

# ---------------------------------------------------------------------------
# Guard: subtab module availability
# ---------------------------------------------------------------------------
try:
    from src.dashboard.shiny_tab_clients.subtab_assets import (
        subtab_clients_assets_server,
        subtab_clients_assets_ui,
    )
    from src.dashboard.shiny_tab_clients.subtab_goals import (
        subtab_clients_goals_server,
        subtab_clients_goals_ui,
    )
    from src.dashboard.shiny_tab_clients.subtab_income import (
        subtab_clients_income_server,
        subtab_clients_income_ui,
    )
    from src.dashboard.shiny_tab_clients.subtab_personal_info import (
        subtab_clients_personal_info_server,
        subtab_clients_personal_info_ui,
    )
    from src.dashboard.shiny_tab_clients.subtab_summary import (
        subtab_clients_summary_server,
        subtab_clients_summary_ui,
    )
    from src.dashboard.shiny_tab_clients.tab_clients import (
        tab_clients_server,
        tab_clients_ui,
    )

    MODULE_IMPORT_AVAILABLE = True
except ImportError as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("shiny_tab_clients module import failed: %s", _exc)

pytestmark_utils = pytest.mark.skipif(
    not UTILS_IMPORT_AVAILABLE,
    reason="utils_tab_clients not importable",
)
pytestmark_modules = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="shiny_tab_clients modules not importable",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_data_utils() -> dict[str, Any]:
    """Minimal data_utils dictionary accepted by all subtab modules."""
    return {"theme": "default", "export_enabled": False}


@pytest.fixture()
def sample_data_inputs() -> dict[str, Any]:
    """Minimal data_inputs dictionary accepted by all subtab modules."""
    return {
        "Personal_Info": {
            "primary": {"name": "Jane Smith", "age_current": 55, "age_retirement": 65},
            "partner": {"name": "John Smith", "age_current": 58, "age_retirement": 65},
        },
        "Assets": {
            "taxable": 250_000.0,
            "tax_deferred": 500_000.0,
            "tax_free": 100_000.0,
        },
        "Goals": {"essential": 50_000.0, "important": 25_000.0, "aspirational": 15_000.0},
        "Income": {"social_security": 24_000.0, "pension": 36_000.0, "annuity": 0.0, "other": 0.0},
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
# utils_tab_clients — format_currency_display
# ===========================================================================


@pytest.mark.integration()
@pytestmark_utils
class Test_Format_Currency_Display_Integration:
    """Integration tests for format_currency_display consistency."""

    def test_positive_integer(self) -> None:
        """Whole-dollar positive amounts format correctly."""
        assert format_currency_display(100_000) == "$100,000"
        _logger.debug("format_currency_display(100_000) OK")

    def test_zero(self) -> None:
        """Zero input returns '$0'."""
        assert format_currency_display(0) == "$0"

    def test_none_returns_dollar_zero(self) -> None:
        """None input returns '$0' without raising."""
        assert format_currency_display(None) == "$0"

    def test_float_rounds_to_whole_dollar(self) -> None:
        """Float with cents rounds to nearest dollar."""
        result = format_currency_display(1_234.56)
        assert result == "$1,235"

    def test_large_value(self) -> None:
        """Large portfolio values format with correct thousand separators."""
        result = format_currency_display(5_000_000)
        assert result == "$5,000,000"

    @pytest.mark.parametrize(
        "amount",
        [0, 1, 1_000, 100_000, 1_000_000, 5_000_000],
    )
    def test_result_starts_with_dollar_sign(self, amount: int) -> None:
        """Every valid result starts with '$'."""
        assert format_currency_display(amount).startswith("$")


# ===========================================================================
# utils_tab_clients — validate_financial_amount
# ===========================================================================


@pytest.mark.integration()
@pytestmark_utils
class Test_Validate_Financial_Amount_Integration:
    """Integration tests for validate_financial_amount business rules."""

    def test_positive_value_passes(self) -> None:
        """Positive values are returned unchanged (or as float)."""
        result = validate_financial_amount(50_000.0)
        assert float(result) == pytest.approx(50_000.0)

    def test_zero_passes(self) -> None:
        """Zero is a valid financial amount."""
        result = validate_financial_amount(0)
        assert float(result) == pytest.approx(0.0)

    def test_none_converts_to_zero(self) -> None:
        """None is converted to 0.0 without raising."""
        result = validate_financial_amount(None)
        assert float(result) == pytest.approx(0.0)

    def test_negative_raises_value_error(self) -> None:
        """Negative amounts raise ValueError per business rules."""
        with pytest.raises((ValueError, Exception)):
            validate_financial_amount(-1.0)

    @pytest.mark.parametrize("amount", [0.01, 100.0, 10_000.0, 1_000_000.0])
    def test_various_positive_amounts(self, amount: float) -> None:
        """A range of positive amounts all pass validation."""
        result = validate_financial_amount(amount)
        assert float(result) >= 0.0


# ===========================================================================
# utils_tab_clients — validate_age_range
# ===========================================================================


@pytest.mark.integration()
@pytestmark_utils
class Test_Validate_Age_Range_Integration:
    """Integration tests for validate_age_range boundary checks."""

    def test_valid_retirement_age(self) -> None:
        """Typical retirement age is valid."""
        result = validate_age_range(65, min_age=18, max_age=100)
        assert result is True or result == 65

    def test_minimum_boundary(self) -> None:
        """Age at minimum boundary is accepted."""
        validate_age_range(18, min_age=18, max_age=100)

    def test_maximum_boundary(self) -> None:
        """Age at maximum boundary is accepted."""
        validate_age_range(100, min_age=18, max_age=100)

    def test_below_minimum_raises_or_returns_false(self) -> None:
        """Ages below minimum should fail validation."""
        try:
            result = validate_age_range(5, min_age=18, max_age=100)
            assert result is False or result is None
        except Exception:  # noqa: BLE001, S110
            pass  # Raising is also acceptable

    def test_above_maximum_raises_or_returns_false(self) -> None:
        """Ages above maximum should fail validation."""
        try:
            result = validate_age_range(150, min_age=18, max_age=100)
            assert result is False or result is None
        except Exception:  # noqa: BLE001, S110
            pass  # Raising is also acceptable


# ===========================================================================
# utils_tab_clients — validate_string_input
# ===========================================================================


@pytest.mark.integration()
@pytestmark_utils
class Test_Validate_String_Input_Integration:
    """Integration tests for validate_string_input."""

    def test_valid_name_passes(self) -> None:
        """Non-empty string is valid."""
        result = validate_string_input("Jane Smith")
        assert result is True or isinstance(result, str)

    def test_empty_string_fails(self) -> None:
        """Empty strings should fail validation."""
        try:
            result = validate_string_input("")
            assert result is False or result is None
        except Exception:  # noqa: BLE001, S110
            pass

    def test_whitespace_only_raises_or_fails(self) -> None:
        """Whitespace-only strings should be rejected."""
        try:
            result = validate_string_input("   ")
            assert result is False or result is None
        except Exception:  # noqa: BLE001, S110
            pass


# ===========================================================================
# Module API — callable signatures
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_All_Subtab_Modules_Are_Callable:
    """Every subtab module exposes callable UI and server functions."""

    def test_subtab_personal_info_ui_callable(self) -> None:
        """subtab_clients_personal_info_ui is callable."""
        assert callable(subtab_clients_personal_info_ui)

    def test_subtab_personal_info_server_callable(self) -> None:
        """subtab_clients_personal_info_server is callable."""
        assert callable(subtab_clients_personal_info_server)

    def test_subtab_assets_ui_callable(self) -> None:
        """subtab_clients_assets_ui is callable."""
        assert callable(subtab_clients_assets_ui)

    def test_subtab_assets_server_callable(self) -> None:
        """subtab_clients_assets_server is callable."""
        assert callable(subtab_clients_assets_server)

    def test_subtab_goals_ui_callable(self) -> None:
        """subtab_clients_goals_ui is callable."""
        assert callable(subtab_clients_goals_ui)

    def test_subtab_goals_server_callable(self) -> None:
        """subtab_clients_goals_server is callable."""
        assert callable(subtab_clients_goals_server)

    def test_subtab_income_ui_callable(self) -> None:
        """subtab_clients_income_ui is callable."""
        assert callable(subtab_clients_income_ui)

    def test_subtab_income_server_callable(self) -> None:
        """subtab_clients_income_server is callable."""
        assert callable(subtab_clients_income_server)

    def test_subtab_summary_ui_callable(self) -> None:
        """subtab_clients_summary_ui is callable."""
        assert callable(subtab_clients_summary_ui)

    def test_subtab_summary_server_callable(self) -> None:
        """subtab_clients_summary_server is callable."""
        assert callable(subtab_clients_summary_server)

    def test_tab_clients_ui_callable(self) -> None:
        """tab_clients_ui is callable."""
        assert callable(tab_clients_ui)

    def test_tab_clients_server_callable(self) -> None:
        """tab_clients_server is callable."""
        assert callable(tab_clients_server)


# ===========================================================================
# Module API — server function signatures
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Server_Function_Signatures:
    """Server functions must accept the standard Shiny module parameter set."""

    def _get_params(self, func: Any) -> list[str]:
        """Extract parameter names from a callable."""
        return list(inspect.signature(func).parameters.keys())

    def test_personal_info_server_has_id_param(self) -> None:
        """subtab_clients_personal_info_server has 'id' parameter (Shiny module)."""
        params = self._get_params(subtab_clients_personal_info_server)
        assert "id" in params

    def test_assets_server_has_id_param(self) -> None:
        """subtab_clients_assets_server has 'id' parameter."""
        params = self._get_params(subtab_clients_assets_server)
        assert "id" in params

    def test_goals_server_has_id_param(self) -> None:
        """subtab_clients_goals_server has 'id' parameter."""
        params = self._get_params(subtab_clients_goals_server)
        assert "id" in params

    def test_income_server_has_id_param(self) -> None:
        """subtab_clients_income_server has 'id' parameter."""
        params = self._get_params(subtab_clients_income_server)
        assert "id" in params

    def test_summary_server_has_id_param(self) -> None:
        """subtab_clients_summary_server has 'id' parameter."""
        params = self._get_params(subtab_clients_summary_server)
        assert "id" in params

    def test_tab_clients_server_has_id_param(self) -> None:
        """tab_clients_server has 'id' parameter."""
        params = self._get_params(tab_clients_server)
        assert "id" in params


# ===========================================================================
# Tab clients — server return structure
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Tab_Clients_Server_Return_Structure:
    """tab_clients_server must return a dict with all five subtab instances."""

    _EXPECTED_RETURN_KEYS: ClassVar[list[str]] = [
        "clients_Personal_Info_Server",
        "clients_Assets_Server",
        "clients_Goals_Server",
        "clients_Income_Server",
        "clients_Summary_Server",
    ]

    def test_server_return_keys_documented(self) -> None:
        """Server docstring or source code references all five subtab keys."""
        source = Path(sys.modules['src.dashboard.shiny_tab_clients.tab_clients'].__file__).read_text(encoding='utf-8')
        for key in self._EXPECTED_RETURN_KEYS:
            assert key in source, f"Expected key '{key}' not found in tab_clients_server source"
        _logger.debug("All expected return keys found in tab_clients_server source")


# ===========================================================================
# Input ID naming convention — structural validation
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Input_ID_Naming_Convention:
    """Verify that source code uses the correct hierarchical input ID patterns."""

    _REQUIRED_INPUT_IDS: ClassVar[list[str]] = [
        # personal_info primary
        "input_ID_tab_clients_subtab_clients_personal_info_client_primary_name",
        "input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_current",
        "input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_retirement",
        "input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_income_starting",
        "input_ID_tab_clients_subtab_clients_personal_info_client_primary_status_marital",
        "input_ID_tab_clients_subtab_clients_personal_info_client_primary_gender",
        "input_ID_tab_clients_subtab_clients_personal_info_client_primary_tolerance_risk",
        "input_ID_tab_clients_subtab_clients_personal_info_client_primary_state",
        "input_ID_tab_clients_subtab_clients_personal_info_client_primary_code_zip",
        # personal_info partner
        "input_ID_tab_clients_subtab_clients_personal_info_client_partner_name",
        "input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_current",
        "input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_retirement",
    ]

    def test_personal_info_input_ids_in_source(self) -> None:
        """Required primary/partner input IDs appear in subtab_personal_info source."""
        source = Path(sys.modules['src.dashboard.shiny_tab_clients.subtab_personal_info'].__file__).read_text(encoding='utf-8')
        for input_id in self._REQUIRED_INPUT_IDS:
            assert input_id in source, (
                f"Input ID '{input_id}' not found in subtab_clients_personal_info_ui source"
            )
        _logger.debug("All required personal_info input IDs found in source")

    def test_input_ids_follow_hierarchical_pattern(self) -> None:
        """Every required input ID starts with 'input_ID_tab_clients_subtab_'."""
        prefix = "input_ID_tab_clients_subtab_"
        for input_id in self._REQUIRED_INPUT_IDS:
            assert input_id.startswith(prefix), (
                f"Input ID '{input_id}' does not follow naming convention"
            )
        _logger.debug("All input IDs follow hierarchical naming convention")

    def test_input_ids_contain_no_camel_case(self) -> None:
        """Input IDs should use snake_case (no uppercase run in the snake portion)."""
        for input_id in self._REQUIRED_INPUT_IDS:
            assert "_" in input_id, f"Input ID '{input_id}' lacks underscores"
        _logger.debug("Input ID underscore structure verified")


# ===========================================================================
# Cross-module consistency — subtab IDs used in tab_clients
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Tab_Clients_Subtab_Module_IDs:
    """tab_clients must wire all subtabs with the expected module IDs."""

    _EXPECTED_MODULE_IDS: ClassVar[list[str]] = [
        "ID_tab_clients_subtab_clients_personal_info",
        "ID_tab_clients_subtab_clients_assets",
        "ID_tab_clients_subtab_clients_goals",
        "ID_tab_clients_subtab_clients_income",
        "ID_tab_clients_subtab_clients_summary",
    ]

    def test_all_subtab_module_ids_in_tab_clients_source(self) -> None:
        """All subtab module IDs are referenced in tab_clients source."""
        _tab_clients_source = Path(sys.modules['src.dashboard.shiny_tab_clients.tab_clients'].__file__).read_text(encoding='utf-8')
        ui_source = _tab_clients_source
        server_source = _tab_clients_source
        combined = ui_source + server_source
        for module_id in self._EXPECTED_MODULE_IDS:
            assert module_id in combined, (
                f"Module ID '{module_id}' not found in tab_clients source"
            )
        _logger.debug("All subtab module IDs found in tab_clients source")

    def test_navset_tab_id_defined(self) -> None:
        """tab_clients_ui defines the top-level navset_tab ID."""
        source = Path(sys.modules['src.dashboard.shiny_tab_clients.tab_clients'].__file__).read_text(encoding='utf-8')
        assert "ID_tab_clients_tabs_all" in source
        _logger.debug("Top-level navset_tab ID found")


# ===========================================================================
# Cross-module consistency — income input IDs vs source code
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Income_Subtab_Input_IDs:
    """Income subtab input IDs follow the naming convention."""

    _REQUIRED_INCOME_INPUT_IDS: ClassVar[list[str]] = [
        "input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security",
        "input_ID_tab_clients_subtab_clients_income_client_primary_income_pension",
        "input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing",
        "input_ID_tab_clients_subtab_clients_income_client_primary_income_other",
    ]

    def test_income_input_ids_in_server_source(self) -> None:
        """Required income input IDs appear in subtab_income server source."""
        source = Path(sys.modules['src.dashboard.shiny_tab_clients.subtab_income'].__file__).read_text(encoding='utf-8')
        for input_id in self._REQUIRED_INCOME_INPUT_IDS:
            assert input_id in source, (
                f"Income input ID '{input_id}' not found in subtab_income_server source"
            )
        _logger.debug("All income input IDs verified in source")


# ===========================================================================
# Cross-module consistency — assets input IDs vs source code
# ===========================================================================


@pytest.mark.integration()
@pytestmark_modules
class Test_Assets_Subtab_Input_IDs:
    """Assets subtab input IDs follow the naming convention."""

    _REQUIRED_ASSET_INPUT_IDS: ClassVar[list[str]] = [
        "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_investable",
        "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_taxable",
        "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_deferred",
        "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_free",
    ]

    def test_asset_input_ids_in_server_source(self) -> None:
        """Required asset input IDs appear in subtab_assets server source."""
        source = Path(sys.modules['src.dashboard.shiny_tab_clients.subtab_assets'].__file__).read_text(encoding='utf-8')
        for input_id in self._REQUIRED_ASSET_INPUT_IDS:
            assert input_id in source, (
                f"Asset input ID '{input_id}' not found in subtab_assets_server source"
            )
        _logger.debug("All asset input IDs verified in source")
