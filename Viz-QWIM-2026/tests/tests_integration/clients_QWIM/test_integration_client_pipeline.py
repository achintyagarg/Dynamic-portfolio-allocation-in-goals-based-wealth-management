"""Integration tests for the client data pipeline.

Verifies that Client_QWIM construction, data population via update methods,
data retrieval, and financial validation utilities work correctly together
as a cohesive client data management pipeline.

Test Categories
---------------
- Client_QWIM construction and initial DataFrame schemas
- update_personal_info populates m_personal_info correctly
- update_assets populates m_assets with correct schema
- update_goals populates m_goals with correct schema
- update_income populates m_income with correct schema
- Chained updates: successive update calls accumulate correctly
- Integration with utils_tab_clients validation functions
- validate_financial_amount → format_currency_display pipeline

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

MODULE_IMPORT_AVAILABLE: bool = True

try:
    from src.clients_QWIM.client_QWIM import (
        Client_QWIM,
        Client_Type,
        Marital_Status,
    )
    from src.dashboard.shiny_utils.utils_tab_clients import (
        format_currency_display,
        validate_age_range,
        validate_financial_amount,
        validate_string_input,
    )
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="Client pipeline modules not importable in this environment",
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture()
def primary_client() -> "Client_QWIM":
    """Provide a freshly constructed primary client object.

    Returns
    -------
    Client_QWIM
        Client instance with no data populated yet.
    """
    return Client_QWIM(
        client_ID="TEST_CLI_001",
        first_name="Jane",
        last_name="Doe",
        client_type=Client_Type.CLIENT_PRIMARY,
    )


@pytest.fixture()
def partner_client() -> "Client_QWIM":
    """Provide a freshly constructed partner client object.

    Returns
    -------
    Client_QWIM
        Partner client instance with no data populated yet.
    """
    return Client_QWIM(
        client_ID="TEST_CLI_002",
        first_name="John",
        last_name="Doe",
        client_type=Client_Type.CLIENT_PARTNER,
    )


@pytest.fixture()
def sample_personal_info() -> dict[str, Any]:
    """Provide sample personal information data for update_personal_info.

    Returns
    -------
    dict[str, Any]
        Dictionary with client personal information keys.
    """
    return {
        "First Name": "Jane",
        "Last Name": "Doe",
        "Age": 45,
        "Retirement Age": 65,
        "Marital Status": Marital_Status.MARRIED.value,
    }


@pytest.fixture()
def sample_assets() -> list[dict[str, Any]]:
    """Provide sample asset data for update_assets.

    Returns
    -------
    list[dict[str, Any]]
        List of asset dictionaries.
    """
    return [
        {
            "Taxable Assets": 250000.0,
            "Tax Deferred Assets": 500000.0,
            "Tax Free Assets": 50000.0,
            "Asset Name": "Brokerage Account",
            "Asset Class": "Stocks",
        }
    ]


@pytest.fixture()
def sample_goals() -> list[dict[str, Any]]:
    """Provide sample financial goals data for update_goals.

    Returns
    -------
    list[dict[str, Any]]
        List of goal dictionaries.
    """
    return [
        {
            "Essential Annual Expense": 60000.0,
            "Important Annual Expense": 20000.0,
            "Aspirational Annual Expense": 15000.0,
            "Essential Annual Expense is Inflation Indexed": True,
            "Important Annual Expense is Inflation Indexed": False,
            "Aspirational Annual Expense is Inflation Indexed": False,
        }
    ]


@pytest.fixture()
def sample_income() -> list[dict[str, Any]]:
    """Provide sample income data for update_income.

    Returns
    -------
    list[dict[str, Any]]
        List of income dictionaries.
    """
    return [
        {
            "Annual Social Security": 24000.0,
        }
    ]


# ==============================================================================
# Client construction integration
# ==============================================================================


@pytest.mark.integration()
class Test_Client_Construction_Integration:
    """Integration tests for Client_QWIM construction and initial state."""

    def test_primary_client_constructs_successfully(
        self, primary_client: "Client_QWIM"
    ) -> None:
        """Client_QWIM constructs without error for CLIENT_PRIMARY type."""
        _logger.debug("Testing Client_QWIM PRIMARY construction")

        assert primary_client is not None, "Client must not be None after construction"

    def test_partner_client_constructs_successfully(
        self, partner_client: "Client_QWIM"
    ) -> None:
        """Client_QWIM constructs without error for CLIENT_PARTNER type."""
        _logger.debug("Testing Client_QWIM PARTNER construction")

        assert partner_client is not None, "Partner client must not be None after construction"

    def test_client_has_non_empty_id_after_construction(
        self, primary_client: "Client_QWIM"
    ) -> None:
        """Client_ID is preserved exactly as passed to the constructor."""
        _logger.debug("Testing Client_QWIM client_ID preservation")

        assert primary_client.m_client_ID == "TEST_CLI_001", (
            "Client ID must match the value passed to the constructor"
        )

    def test_assets_dataframe_initialised_with_correct_schema(
        self, primary_client: "Client_QWIM"
    ) -> None:
        """m_assets is a Polars DataFrame with expected numeric asset columns."""
        _logger.debug("Testing m_assets initialised as DataFame")

        assets_df = primary_client.m_assets

        assert isinstance(assets_df, pl.DataFrame), "m_assets must be a Polars DataFrame"
        expected_cols = ["Taxable Assets", "Tax Deferred Assets", "Tax Free Assets"]
        for col in expected_cols:
            assert col in assets_df.columns, f"m_assets must contain '{col}' column"

    def test_goals_dataframe_initialised_with_correct_schema(
        self, primary_client: "Client_QWIM"
    ) -> None:
        """m_goals is a Polars DataFrame with expected expense columns."""
        _logger.debug("Testing m_goals initialised as DataFrame")

        goals_df = primary_client.m_goals

        assert isinstance(goals_df, pl.DataFrame), "m_goals must be a Polars DataFrame"
        expected_cols = [
            "Essential Annual Expense",
            "Important Annual Expense",
            "Aspirational Annual Expense",
        ]
        for col in expected_cols:
            assert col in goals_df.columns, f"m_goals must contain '{col}' column"

    def test_income_dataframe_initialised(
        self, primary_client: "Client_QWIM"
    ) -> None:
        """m_income is a Polars DataFrame after construction."""
        _logger.debug("Testing m_income is a Polars DataFrame after construction")

        assert isinstance(primary_client.m_income, pl.DataFrame), (
            "m_income must be a Polars DataFrame after construction"
        )


# ==============================================================================
# update_assets integration
# ==============================================================================


@pytest.mark.integration()
class Test_Client_Update_Assets_Integration:
    """Integration tests for update_assets populating m_assets DataFrame."""

    def test_update_assets_returns_true(
        self,
        primary_client: "Client_QWIM",
        sample_assets: list[dict[str, Any]],
    ) -> None:
        """update_assets returns True when given valid asset data."""
        _logger.debug("Testing update_assets return value with valid data")

        result = primary_client.update_assets(sample_assets)

        assert result is True, "update_assets must return True for valid input"

    def test_update_assets_populates_dataframe(
        self,
        primary_client: "Client_QWIM",
        sample_assets: list[dict[str, Any]],
    ) -> None:
        """m_assets contains one row after update_assets with one asset dict."""
        _logger.debug("Testing m_assets has 1 row after update_assets")

        primary_client.update_assets(sample_assets)
        assets_df = primary_client.m_assets

        assert isinstance(assets_df, pl.DataFrame), "m_assets must be a Polars DataFrame"
        assert assets_df.height == 1, f"m_assets must have 1 row, got {assets_df.height}"

    def test_update_assets_taxable_value_is_correct(
        self,
        primary_client: "Client_QWIM",
        sample_assets: list[dict[str, Any]],
    ) -> None:
        """Taxable Assets value is stored correctly after update_assets."""
        _logger.debug("Testing Taxable Assets value after update_assets")

        primary_client.update_assets(sample_assets)
        assets_df = primary_client.m_assets

        assert "Taxable Assets" in assets_df.columns, "m_assets must have 'Taxable Assets' column"
        taxable_value = assets_df["Taxable Assets"][0]
        assert abs(taxable_value - 250000.0) < 0.01, (
            f"Taxable Assets must be 250000.0, got {taxable_value}"
        )

    def test_update_assets_with_empty_list_resets_to_empty_dataframe(
        self, primary_client: "Client_QWIM"
    ) -> None:
        """update_assets with empty list resets m_assets to an empty DataFrame."""
        _logger.debug("Testing update_assets resets m_assets with empty list")

        result = primary_client.update_assets([])

        assert result is True, "update_assets with empty list must return True"
        assert isinstance(primary_client.m_assets, pl.DataFrame), "m_assets must still be DataFrame"
        assert primary_client.m_assets.height == 0, "m_assets must be empty after update with []"


# ==============================================================================
# update_goals integration
# ==============================================================================


@pytest.mark.integration()
class Test_Client_Update_Goals_Integration:
    """Integration tests for update_goals populating m_goals DataFrame."""

    def test_update_goals_returns_true(
        self,
        primary_client: "Client_QWIM",
        sample_goals: list[dict[str, Any]],
    ) -> None:
        """update_goals returns True for valid goal data."""
        _logger.debug("Testing update_goals return value with valid data")

        result = primary_client.update_goals(sample_goals)

        assert result is True, "update_goals must return True for valid input"

    def test_update_goals_populates_dataframe(
        self,
        primary_client: "Client_QWIM",
        sample_goals: list[dict[str, Any]],
    ) -> None:
        """m_goals contains one row after update_goals with one goal dict."""
        _logger.debug("Testing m_goals has 1 row after update_goals")

        primary_client.update_goals(sample_goals)

        assert primary_client.m_goals.height == 1, (
            f"m_goals must have 1 row, got {primary_client.m_goals.height}"
        )

    def test_update_goals_essential_expense_value_is_correct(
        self,
        primary_client: "Client_QWIM",
        sample_goals: list[dict[str, Any]],
    ) -> None:
        """Essential Annual Expense is stored correctly after update_goals."""
        _logger.debug("Testing Essential Annual Expense value after update_goals")

        primary_client.update_goals(sample_goals)
        goals_df = primary_client.m_goals

        assert "Essential Annual Expense" in goals_df.columns
        essential = goals_df["Essential Annual Expense"][0]
        assert abs(essential - 60000.0) < 0.01, (
            f"Essential Annual Expense must be 60000.0, got {essential}"
        )


# ==============================================================================
# update_personal_info integration
# ==============================================================================


@pytest.mark.integration()
class Test_Client_Update_Personal_Info_Integration:
    """Integration tests for update_personal_info populating m_personal_info."""

    def test_update_personal_info_returns_true(
        self,
        primary_client: "Client_QWIM",
        sample_personal_info: dict[str, Any],
    ) -> None:
        """update_personal_info returns True for valid personal info dict."""
        _logger.debug("Testing update_personal_info return value")

        result = primary_client.update_personal_info(sample_personal_info)

        assert result is True, "update_personal_info must return True for valid input"

    def test_update_personal_info_first_name_stored(
        self,
        primary_client: "Client_QWIM",
        sample_personal_info: dict[str, Any],
    ) -> None:
        """First Name from personal info dict is stored on the client object."""
        _logger.debug("Testing First Name stored after update_personal_info")

        primary_client.update_personal_info(sample_personal_info)

        assert primary_client.m_first_name == "Jane", (
            f"First Name must be 'Jane', got '{primary_client.m_first_name}'"
        )

    def test_update_personal_info_last_name_stored(
        self,
        primary_client: "Client_QWIM",
        sample_personal_info: dict[str, Any],
    ) -> None:
        """Last Name from personal info dict is stored on the client object."""
        _logger.debug("Testing Last Name stored after update_personal_info")

        primary_client.update_personal_info(sample_personal_info)

        assert primary_client.m_last_name == "Doe", (
            f"Last Name must be 'Doe', got '{primary_client.m_last_name}'"
        )


# ==============================================================================
# validate_financial_amount + format_currency_display integration
# ==============================================================================


@pytest.mark.integration()
class Test_Financial_Validation_And_Formatting_Integration:
    """Integration tests for validate_financial_amount → format_currency_display pipeline."""

    @pytest.mark.parametrize(
        "raw_amount, expected_display",
        [
            (100000.0, "$100,000"),
            (500000.49, "$500,000"),
            (0.0, "$0"),
            (1_000_000.0, "$1,000,000"),
        ],
    )
    def test_validate_then_format_pipeline(
        self, raw_amount: float, expected_display: str
    ) -> None:
        """Validated amount formats correctly through the display pipeline."""
        _logger.debug("Testing validate %s -> format pipeline", raw_amount)

        validated = validate_financial_amount(raw_amount)
        display = format_currency_display(validated)

        assert display == expected_display, (
            f"Amount {raw_amount} must display as '{expected_display}', got '{display}'"
        )

    def test_validate_financial_amount_rejects_negative(self) -> None:
        """validate_financial_amount raises ValueError for negative amounts."""
        _logger.debug("Testing validate_financial_amount rejects negative")

        with pytest.raises(ValueError):
            validate_financial_amount(-1.0)

    def test_format_currency_display_with_none_returns_zero(self) -> None:
        """format_currency_display returns '$0' for None input."""
        _logger.debug("Testing format_currency_display handles None")

        result = format_currency_display(None)

        assert result == "$0", f"None input must format as '$0', got '{result}'"

    def test_validate_age_range_returns_integer(self) -> None:
        """validate_age_range returns the validated age as an integer."""
        _logger.debug("Testing validate_age_range returns integer")

        result = validate_age_range(45, 18, 100, "Test age")

        assert isinstance(result, int), f"validate_age_range must return int, got {type(result)}"
        assert result == 45, f"Validated age must be 45, got {result}"

    def test_validate_age_range_raises_for_out_of_range(self) -> None:
        """validate_age_range raises an error when age is outside min/max bounds."""
        _logger.debug("Testing validate_age_range rejects out-of-range age")

        with pytest.raises((ValueError, Exception)):
            validate_age_range(17, 18, 100, "Test age under minimum")

    def test_validate_string_input_strips_whitespace(self) -> None:
        """validate_string_input returns a stripped string."""
        _logger.debug("Testing validate_string_input strips whitespace")

        result = validate_string_input("  Jane Doe  ", "test field")

        assert result == "Jane Doe", f"Stripped string must be 'Jane Doe', got '{result}'"


# ==============================================================================
# Full client lifecycle integration
# ==============================================================================


@pytest.mark.integration()
class Test_Client_Lifecycle_Integration:
    """Integration tests verifying the complete client data lifecycle."""

    def test_full_client_population_pipeline(
        self,
        primary_client: "Client_QWIM",
        sample_personal_info: dict[str, Any],
        sample_assets: list[dict[str, Any]],
        sample_goals: list[dict[str, Any]],
        sample_income: list[dict[str, Any]],
    ) -> None:
        """Full pipeline: construct → update all fields → retrieve data produces valid state."""
        _logger.debug("Testing complete Client_QWIM population pipeline")

        result_personal = primary_client.update_personal_info(sample_personal_info)
        result_assets = primary_client.update_assets(sample_assets)
        result_goals = primary_client.update_goals(sample_goals)
        result_income = primary_client.update_income(sample_income)

        assert result_personal is True, "update_personal_info must succeed"
        assert result_assets is True, "update_assets must succeed"
        assert result_goals is True, "update_goals must succeed"
        assert result_income is True, "update_income must succeed"

        assert primary_client.m_first_name == "Jane", "First name must be Jane"
        assert primary_client.m_assets.height == 1, "Must have 1 asset row"
        assert primary_client.m_goals.height == 1, "Must have 1 goals row"
        assert isinstance(primary_client.m_income, pl.DataFrame), "Income must be DataFrame"

    def test_get_personal_info_returns_dataframe(
        self, primary_client: "Client_QWIM"
    ) -> None:
        """get_personal_info() returns the client's personal info as a Polars DataFrame."""
        _logger.debug("Testing get_personal_info return type")

        personal_info = primary_client.get_personal_info()

        assert isinstance(personal_info, pl.DataFrame), (
            "get_personal_info() must return a Polars DataFrame"
        )

    def test_successive_update_assets_replaces_data(
        self,
        primary_client: "Client_QWIM",
        sample_assets: list[dict[str, Any]],
    ) -> None:
        """Calling update_assets a second time with different data replaces the first."""
        _logger.debug("Testing successive update_assets call replaces prior data")

        primary_client.update_assets(sample_assets)
        assert primary_client.m_assets.height == 1, "After first update_assets: should have 1 row"

        second_assets = [
            {
                "Taxable Assets": 100000.0,
                "Tax Deferred Assets": 200000.0,
                "Tax Free Assets": 20000.0,
                "Asset Name": "Second Account",
                "Asset Class": "Bonds",
            },
            {
                "Taxable Assets": 50000.0,
                "Tax Deferred Assets": 0.0,
                "Tax Free Assets": 0.0,
                "Asset Name": "Third Account",
                "Asset Class": "Cash",
            },
        ]
        primary_client.update_assets(second_assets)

        assert primary_client.m_assets.height == 2, (
            f"After second update_assets with 2 records: should have 2 rows, "
            f"got {primary_client.m_assets.height}"
        )
