"""Unit and integration tests for ``src.dashboard.shiny_utils.utils_reporting``.

Covers:

* ``validate_data_clients_in_reactives``  /  ``validate_client_level_name``
  /  ``validate_client_data_category_name``
* ``get_single_or_couple_from_reactives``  /  ``update_single_or_couple_in_reactives``
* ``get_client_data_from_reactives``  /  ``save_client_data_to_reactives``
* Convenience client wrappers (Personal_Info, Assets, Goals, Income, Combined)
* ``build_client_info_json_from_reactives``
* ``validate_data_results_in_reactives``  /  ``validate_results_subtab_key``
* ``get_results_data_from_reactives``  /  ``save_results_data_to_reactives``
* Convenience results wrappers (10 subtab save functions)
* ``_coerce_results_value_to_json``  /  ``build_results_data_json_from_reactives``

All tests that require a reactive context use ``shiny.reactive.isolate()``.

Run:
    pytest tests/tests_shiny/ -m "unit or regression" -k utils_reporting
"""

from __future__ import annotations

from typing import Any

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def reactives_with_data_clients(data_utils: dict[str, Any]) -> dict[str, Any]:
    """Return a fully initialised reactives_shiny dict (module scope for speed)."""
    from shiny import reactive

    from src.dashboard.shiny_utils.reactives_shiny import initialize_reactives_shiny

    with reactive.isolate():
        return initialize_reactives_shiny(data_utils=data_utils)


@pytest.fixture()
def sample_personal_info_df() -> Any:
    """Return a minimal Personal_Info Polars DataFrame."""
    import polars as pl

    return pl.DataFrame(
        {
            "name": ["Jane Doe"],
            "age_current": [45],
            "age_retirement": [65],
            "age_income_starting": [67],
            "status_marital": ["Married"],
            "gender": ["Female"],
            "tolerance_risk": ["Moderate"],
            "state": ["CA"],
            "code_zip": ["90210"],
        },
    )


@pytest.fixture()
def sample_assets_df() -> Any:
    """Return a minimal Assets Polars DataFrame."""
    import polars as pl

    return pl.DataFrame(
        {
            "taxable": [250_000.0],
            "tax_deferred": [500_000.0],
            "tax_free": [75_000.0],
            "total": [825_000.0],
        },
    )


@pytest.fixture()
def sample_goals_df() -> Any:
    """Return a minimal Goals Polars DataFrame."""
    import polars as pl

    return pl.DataFrame(
        {
            "essential": [1_500_000.0],
            "important": [500_000.0],
            "aspirational": [250_000.0],
            "total": [2_250_000.0],
        },
    )


@pytest.fixture()
def sample_income_df() -> Any:
    """Return a minimal Income Polars DataFrame."""
    import polars as pl

    return pl.DataFrame(
        {
            "social_security": [24_000.0],
            "pension": [12_000.0],
            "annuity_existing": [0.0],
            "other": [0.0],
            "total": [36_000.0],
        },
    )


# =============================================================================
# validate_data_clients_in_reactives
# =============================================================================


@pytest.mark.unit()
class TestValidateDataClientsInReactives:
    """``validate_data_clients_in_reactives`` checks the Data_Clients schema."""

    def test_none_returns_false_or_none(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_data_clients_in_reactives

        result = validate_data_clients_in_reactives(None)  # type: ignore[arg-type]
        # Early-return None or (False, msg) are both acceptable per project convention
        assert result is None or (isinstance(result, tuple) and result[0] is False)

    def test_non_dict_returns_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_data_clients_in_reactives

        ok, _ = validate_data_clients_in_reactives("not-a-dict")  # type: ignore[arg-type]
        assert ok is False

    def test_missing_data_clients_key_returns_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_data_clients_in_reactives

        ok, msg = validate_data_clients_in_reactives({"Wrong": {}})
        assert ok is False
        assert "Data_Clients" in msg

    def test_data_clients_not_dict_returns_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_data_clients_in_reactives

        ok, _msg = validate_data_clients_in_reactives({"Data_Clients": "bad"})
        assert ok is False

    def test_missing_single_or_couple_returns_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_data_clients_in_reactives

        ok, msg = validate_data_clients_in_reactives(
            {
                "Data_Clients": {
                    "Client_Primary": {},
                    "Client_Partner": {},
                    "Clients_Combined": {},
                },
            },
        )
        assert ok is False
        assert "Single_Or_Couple" in msg

    @pytest.mark.parametrize(
        "missing_sub_level",
        ["Client_Primary", "Client_Partner", "Clients_Combined"],
    )
    def test_missing_sub_level_returns_false(self, missing_sub_level: str) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_data_clients_in_reactives

        data_clients: dict[str, Any] = {
            "Single_Or_Couple": None,
            "Client_Primary": {},
            "Client_Partner": {},
            "Clients_Combined": {},
        }
        del data_clients[missing_sub_level]
        ok, _ = validate_data_clients_in_reactives({"Data_Clients": data_clients})
        assert ok is False

    def test_valid_structure_returns_true(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_data_clients_in_reactives

        ok, msg = validate_data_clients_in_reactives(reactives_with_data_clients)
        assert ok is True
        assert isinstance(msg, str)


# =============================================================================
# validate_client_level_name
# =============================================================================


@pytest.mark.unit()
class TestValidateClientLevelName:
    """``validate_client_level_name`` accepts exactly the three valid sub-levels."""

    @pytest.mark.parametrize(
        "valid_level",
        ["Client_Primary", "Client_Partner", "Clients_Combined"],
    )
    def test_valid_levels(self, valid_level: str) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_client_level_name

        ok, msg = validate_client_level_name(valid_level)
        assert ok is True
        assert isinstance(msg, str)

    def test_none_returns_none_or_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_client_level_name

        result = validate_client_level_name(None)  # type: ignore[arg-type]
        assert result is None or (isinstance(result, tuple) and result[0] is False)

    def test_non_string_returns_none_or_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_client_level_name

        result = validate_client_level_name(123)  # type: ignore[arg-type]
        assert result is None or (isinstance(result, tuple) and result[0] is False)

    def test_empty_string_returns_none_or_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_client_level_name

        result = validate_client_level_name("")
        assert result is None or (isinstance(result, tuple) and result[0] is False)

    def test_unknown_level_returns_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_client_level_name

        ok, msg = validate_client_level_name("Bad_Level")
        assert ok is False
        assert "Bad_Level" in msg

    def test_case_sensitive(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_client_level_name

        ok, _ = validate_client_level_name("client_primary")
        assert ok is False


# =============================================================================
# validate_client_data_category_name
# =============================================================================


@pytest.mark.unit()
class TestValidateClientDataCategoryName:
    """``validate_client_data_category_name`` applies level-specific restrictions."""

    @pytest.mark.parametrize(
        "category",
        ["Personal_Info", "Assets", "Goals", "Income"],
    )
    def test_valid_categories_for_primary(self, category: str) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            validate_client_data_category_name,
        )

        ok, _ = validate_client_data_category_name(category, "Client_Primary")
        assert ok is True

    def test_personal_info_invalid_for_combined(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            validate_client_data_category_name,
        )

        ok, msg = validate_client_data_category_name("Personal_Info", "Clients_Combined")
        assert ok is False
        assert "Personal_Info" in msg or "Clients_Combined" in msg

    @pytest.mark.parametrize("category", ["Assets", "Goals", "Income"])
    def test_shared_categories_valid_for_combined(self, category: str) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            validate_client_data_category_name,
        )

        ok, _ = validate_client_data_category_name(category, "Clients_Combined")
        assert ok is True

    def test_none_category_returns_none_or_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            validate_client_data_category_name,
        )

        result = validate_client_data_category_name(None)  # type: ignore[arg-type]
        assert result is None or (isinstance(result, tuple) and result[0] is False)

    def test_unknown_category_returns_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            validate_client_data_category_name,
        )

        ok, msg = validate_client_data_category_name("Bad_Category")
        assert ok is False
        assert "Bad_Category" in msg


# =============================================================================
# get_single_or_couple_from_reactives / update_single_or_couple_in_reactives
# =============================================================================


@pytest.mark.unit()
class TestSingleOrCouple:
    """Round-trip tests for Single_Or_Couple in Data_Clients."""

    def test_default_value_is_single_or_couple(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_single_or_couple_from_reactives,
        )

        with reactive.isolate():
            value = get_single_or_couple_from_reactives(reactives_with_data_clients)
        assert value in ("Single", "Couple")

    @pytest.mark.parametrize("value", ["Single", "Couple"])
    def test_roundtrip_set_get(
        self, reactives_with_data_clients: dict[str, Any], value: str,
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_single_or_couple_from_reactives,
            update_single_or_couple_in_reactives,
        )

        with reactive.isolate():
            update_single_or_couple_in_reactives(reactives_with_data_clients, value)
            result = get_single_or_couple_from_reactives(reactives_with_data_clients)
        assert result == value

    def test_invalid_value_raises_value_error(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            update_single_or_couple_in_reactives,
        )

        with pytest.raises(ValueError, match="Invalid single_or_couple"), reactive.isolate():
            update_single_or_couple_in_reactives(
                reactives_with_data_clients, "Invalid",
            )

    def test_none_value_raises_value_error(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            update_single_or_couple_in_reactives,
        )

        with pytest.raises(ValueError), reactive.isolate():
            update_single_or_couple_in_reactives(
                reactives_with_data_clients, None,  # type: ignore[arg-type]
            )

    def test_invalid_reactives_raises_value_error(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_single_or_couple_from_reactives,
        )

        with pytest.raises(ValueError), reactive.isolate():
            get_single_or_couple_from_reactives({"bad": "structure"})


# =============================================================================
# get_client_data_from_reactives / save_client_data_to_reactives (core)
# =============================================================================


@pytest.mark.unit()
class TestGetSaveClientData:
    """Core get/save round-trip for client data."""

    @pytest.mark.parametrize(
        "client_level,category",
        [
            ("Client_Primary", "Personal_Info"),
            ("Client_Primary", "Assets"),
            ("Client_Primary", "Goals"),
            ("Client_Primary", "Income"),
            ("Client_Partner", "Personal_Info"),
            ("Client_Partner", "Assets"),
            ("Clients_Combined", "Assets"),
            ("Clients_Combined", "Goals"),
            ("Clients_Combined", "Income"),
        ],
    )
    def test_save_and_get_none_clears_value(
        self,
        reactives_with_data_clients: dict[str, Any],
        client_level: str,
        category: str,
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_client_data_from_reactives,
            save_client_data_to_reactives,
        )

        with reactive.isolate():
            save_client_data_to_reactives(
                reactives_with_data_clients, client_level, category, None,
            )
            result = get_client_data_from_reactives(
                reactives_with_data_clients, client_level, category,
            )
        assert result is None

    def test_save_and_get_dataframe(
        self,
        reactives_with_data_clients: dict[str, Any],
        sample_assets_df: Any,
    ) -> None:
        import polars as pl

        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_client_data_from_reactives,
            save_client_data_to_reactives,
        )

        with reactive.isolate():
            save_client_data_to_reactives(
                reactives_with_data_clients, "Client_Primary", "Assets", sample_assets_df,
            )
            result = get_client_data_from_reactives(
                reactives_with_data_clients, "Client_Primary", "Assets",
            )
        assert isinstance(result, pl.DataFrame)
        assert result.shape == sample_assets_df.shape

    def test_invalid_level_raises_value_error(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import save_client_data_to_reactives

        with pytest.raises(ValueError), reactive.isolate():
            save_client_data_to_reactives(
                reactives_with_data_clients, "Bad_Level", "Assets", None,
            )

    def test_personal_info_for_combined_raises_value_error(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import save_client_data_to_reactives

        with pytest.raises(ValueError), reactive.isolate():
            save_client_data_to_reactives(
                reactives_with_data_clients, "Clients_Combined", "Personal_Info", None,
            )

    def test_non_dataframe_data_value_raises_value_error(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import save_client_data_to_reactives

        with pytest.raises(ValueError), reactive.isolate():
            save_client_data_to_reactives(
                reactives_with_data_clients,
                "Client_Primary",
                "Assets",
                {"not": "a dataframe"},  # type: ignore[arg-type]
            )

    def test_save_returns_reactives_dict(
        self,
        reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import save_client_data_to_reactives

        with reactive.isolate():
            result = save_client_data_to_reactives(
                reactives_with_data_clients, "Client_Primary", "Assets", None,
            )
        assert isinstance(result, dict)


# =============================================================================
# Convenience client wrappers
# =============================================================================


@pytest.mark.unit()
class TestConvenienceClientWrappers:
    """Convenience wrappers delegate correctly and enforce their restriction."""

    def test_save_personal_info_primary_roundtrip(
        self,
        reactives_with_data_clients: dict[str, Any],
        sample_personal_info_df: Any,
    ) -> None:
        import polars as pl

        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_client_data_from_reactives,
            save_client_personal_info_to_reactives,
        )

        with reactive.isolate():
            save_client_personal_info_to_reactives(
                reactives_with_data_clients, "Client_Primary", sample_personal_info_df,
            )
            result = get_client_data_from_reactives(
                reactives_with_data_clients, "Client_Primary", "Personal_Info",
            )
        assert isinstance(result, pl.DataFrame)

    def test_save_personal_info_to_combined_raises(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            save_client_personal_info_to_reactives,
        )

        with pytest.raises(ValueError), reactive.isolate():
            save_client_personal_info_to_reactives(
                reactives_with_data_clients, "Clients_Combined", None,
            )

    def test_save_assets_to_combined(
        self,
        reactives_with_data_clients: dict[str, Any],
        sample_assets_df: Any,
    ) -> None:
        import polars as pl

        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_client_data_from_reactives,
            save_client_assets_to_reactives,
        )

        with reactive.isolate():
            save_client_assets_to_reactives(
                reactives_with_data_clients, "Clients_Combined", sample_assets_df,
            )
            result = get_client_data_from_reactives(
                reactives_with_data_clients, "Clients_Combined", "Assets",
            )
        assert isinstance(result, pl.DataFrame)

    def test_save_goals_roundtrip(
        self,
        reactives_with_data_clients: dict[str, Any],
        sample_goals_df: Any,
    ) -> None:
        import polars as pl

        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_client_data_from_reactives,
            save_client_goals_to_reactives,
        )

        with reactive.isolate():
            save_client_goals_to_reactives(
                reactives_with_data_clients, "Client_Partner", sample_goals_df,
            )
            result = get_client_data_from_reactives(
                reactives_with_data_clients, "Client_Partner", "Goals",
            )
        assert isinstance(result, pl.DataFrame)

    def test_save_income_roundtrip(
        self,
        reactives_with_data_clients: dict[str, Any],
        sample_income_df: Any,
    ) -> None:
        import polars as pl

        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_client_data_from_reactives,
            save_client_income_to_reactives,
        )

        with reactive.isolate():
            save_client_income_to_reactives(
                reactives_with_data_clients, "Client_Primary", sample_income_df,
            )
            result = get_client_data_from_reactives(
                reactives_with_data_clients, "Client_Primary", "Income",
            )
        assert isinstance(result, pl.DataFrame)

    def test_save_clients_combined_data(
        self,
        reactives_with_data_clients: dict[str, Any],
        sample_income_df: Any,
    ) -> None:
        import polars as pl

        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_client_data_from_reactives,
            save_clients_combined_data_to_reactives,
        )

        with reactive.isolate():
            save_clients_combined_data_to_reactives(
                reactives_with_data_clients, "Income", sample_income_df,
            )
            result = get_client_data_from_reactives(
                reactives_with_data_clients, "Clients_Combined", "Income",
            )
        assert isinstance(result, pl.DataFrame)

    def test_save_clients_combined_personal_info_raises(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            save_clients_combined_data_to_reactives,
        )

        with pytest.raises(ValueError), reactive.isolate():
            save_clients_combined_data_to_reactives(
                reactives_with_data_clients, "Personal_Info", None,
            )


# =============================================================================
# build_client_info_json_from_reactives
# =============================================================================


@pytest.mark.unit()
class TestBuildClientInfoJsonFromReactives:
    """``build_client_info_json_from_reactives`` must return a well-formed dict."""

    def test_returns_dict(self, reactives_with_data_clients: dict[str, Any]) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_client_info_json_from_reactives,
        )

        with reactive.isolate():
            result = build_client_info_json_from_reactives(reactives_with_data_clients)
        assert isinstance(result, dict)

    def test_top_level_keys(self, reactives_with_data_clients: dict[str, Any]) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_client_info_json_from_reactives,
        )

        with reactive.isolate():
            result = build_client_info_json_from_reactives(reactives_with_data_clients)
        assert set(result.keys()) == {
            "single_or_couple",
            "personal_info",
            "assets",
            "goals",
            "income",
        }

    def test_single_or_couple_value(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_client_info_json_from_reactives,
        )

        with reactive.isolate():
            result = build_client_info_json_from_reactives(reactives_with_data_clients)
        assert result["single_or_couple"] in ("Single", "Couple")

    def test_personal_info_has_primary_and_partner(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_client_info_json_from_reactives,
        )

        with reactive.isolate():
            result = build_client_info_json_from_reactives(reactives_with_data_clients)
        assert "primary" in result["personal_info"]
        assert "partner" in result["personal_info"]

    def test_assets_has_primary_partner_combined(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_client_info_json_from_reactives,
        )

        with reactive.isolate():
            result = build_client_info_json_from_reactives(reactives_with_data_clients)
        for key in ("primary", "partner", "combined"):
            assert key in result["assets"], f"Missing assets key: {key}"


# =============================================================================
# validate_data_results_in_reactives
# =============================================================================


@pytest.mark.unit()
class TestValidateDataResultsInReactives:
    """``validate_data_results_in_reactives`` checks the Data_Results schema."""

    def test_none_returns_none_or_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            validate_data_results_in_reactives,
        )

        result = validate_data_results_in_reactives(None)  # type: ignore[arg-type]
        assert result is None or (isinstance(result, tuple) and result[0] is False)

    def test_missing_data_results_key_returns_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            validate_data_results_in_reactives,
        )

        ok, msg = validate_data_results_in_reactives({"Other": {}})
        assert ok is False
        assert "Data_Results" in msg

    def test_valid_structure_returns_true(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            validate_data_results_in_reactives,
        )

        # The full reactives_shiny includes Data_Results
        ok, msg = validate_data_results_in_reactives(reactives_with_data_clients)
        assert ok is True
        assert isinstance(msg, str)

    def test_data_results_not_dict_returns_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            validate_data_results_in_reactives,
        )

        ok, _ = validate_data_results_in_reactives({"Data_Results": "not_a_dict"})
        assert ok is False

    @pytest.mark.parametrize(
        "missing_key",
        [
            "Portfolio_Analysis_Inputs",
            "Portfolio_Analysis_Outputs",
            "Portfolio_Simulation_Inputs",
            "Portfolio_Simulation_Outputs",
            "Portfolio_Optimization_Skfolio_Inputs",
        ],
    )
    def test_missing_subtab_key_returns_false(self, missing_key: str) -> None:
        from src.dashboard.shiny_utils.utils_reporting import (
            VALID_RESULTS_SUBTAB_KEYS,
            validate_data_results_in_reactives,
        )

        incomplete = {k: None for k in VALID_RESULTS_SUBTAB_KEYS if k != missing_key}
        ok, _msg = validate_data_results_in_reactives({"Data_Results": incomplete})
        assert ok is False


# =============================================================================
# validate_results_subtab_key
# =============================================================================


@pytest.mark.unit()
class TestValidateResultsSubtabKey:
    """``validate_results_subtab_key`` accepts only the ten valid keys."""

    @pytest.mark.parametrize(
        "valid_key",
        [
            "Portfolio_Analysis_Inputs",
            "Portfolio_Analysis_Outputs",
            "Portfolio_Comparison_Inputs",
            "Portfolio_Comparison_Outputs",
            "Weights_Analysis_Inputs",
            "Weights_Analysis_Outputs",
            "Portfolio_Optimization_Skfolio_Inputs",
            "Portfolio_Optimization_Skfolio_Outputs",
            "Portfolio_Simulation_Inputs",
            "Portfolio_Simulation_Outputs",
        ],
    )
    def test_valid_keys(self, valid_key: str) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_results_subtab_key

        ok, msg = validate_results_subtab_key(valid_key)
        assert ok is True
        assert isinstance(msg, str)

    def test_none_returns_none_or_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_results_subtab_key

        result = validate_results_subtab_key(None)  # type: ignore[arg-type]
        assert result is None or (isinstance(result, tuple) and result[0] is False)

    def test_empty_string_returns_none_or_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_results_subtab_key

        result = validate_results_subtab_key("")
        assert result is None or (isinstance(result, tuple) and result[0] is False)

    def test_unknown_key_returns_false(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_results_subtab_key

        ok, msg = validate_results_subtab_key("Unknown_Subtab")
        assert ok is False
        assert "Unknown_Subtab" in msg

    def test_case_sensitive(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import validate_results_subtab_key

        ok, _ = validate_results_subtab_key("portfolio_analysis_inputs")
        assert ok is False


# =============================================================================
# get_results_data_from_reactives / save_results_data_to_reactives
# =============================================================================


@pytest.mark.unit()
class TestGetSaveResultsData:
    """Core get/save round-trip for Data_Results values."""

    @pytest.mark.parametrize(
        "subtab_key",
        [
            "Portfolio_Analysis_Inputs",
            "Portfolio_Comparison_Outputs",
            "Weights_Analysis_Inputs",
            "Portfolio_Optimization_Skfolio_Outputs",
            "Portfolio_Simulation_Inputs",
        ],
    )
    def test_initial_value_is_none(
        self,
        reactives_with_data_clients: dict[str, Any],
        subtab_key: str,
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import get_results_data_from_reactives

        with reactive.isolate():
            result = get_results_data_from_reactives(
                reactives_with_data_clients, subtab_key,
            )
        # Either None or some value — initial value after init is None
        assert result is None

    def test_save_and_get_dict_roundtrip(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_results_data_from_reactives,
            save_results_data_to_reactives,
        )

        payload = {"metric": "sharpe", "value": 1.45}
        with reactive.isolate():
            save_results_data_to_reactives(
                reactives_with_data_clients,
                "Portfolio_Analysis_Outputs",
                payload,
            )
            result = get_results_data_from_reactives(
                reactives_with_data_clients, "Portfolio_Analysis_Outputs",
            )
        assert result == payload

    def test_save_dataframe_roundtrip(
        self,
        reactives_with_data_clients: dict[str, Any],
        sample_assets_df: Any,
    ) -> None:
        import polars as pl

        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            get_results_data_from_reactives,
            save_results_data_to_reactives,
        )

        with reactive.isolate():
            save_results_data_to_reactives(
                reactives_with_data_clients,
                "Weights_Analysis_Outputs",
                sample_assets_df,
            )
            result = get_results_data_from_reactives(
                reactives_with_data_clients, "Weights_Analysis_Outputs",
            )
        assert isinstance(result, pl.DataFrame)

    def test_invalid_subtab_key_raises_value_error(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            save_results_data_to_reactives,
        )

        with pytest.raises(ValueError), reactive.isolate():
            save_results_data_to_reactives(
                reactives_with_data_clients, "Invalid_Key", None,
            )

    def test_save_returns_reactives_dict(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import save_results_data_to_reactives

        with reactive.isolate():
            result = save_results_data_to_reactives(
                reactives_with_data_clients,
                "Portfolio_Simulation_Inputs",
                {"distribution_type": "normal"},
            )
        assert isinstance(result, dict)


# =============================================================================
# Convenience results wrappers
# =============================================================================


@pytest.mark.unit()
class TestConvenienceResultsWrappers:
    """Each convenience wrapper must call the correct Data_Results subtab key."""

    @pytest.mark.parametrize(
        "wrapper_name,subtab_key",
        [
            ("save_portfolio_analysis_inputs_to_reactives", "Portfolio_Analysis_Inputs"),
            ("save_portfolio_analysis_outputs_to_reactives", "Portfolio_Analysis_Outputs"),
            ("save_portfolio_comparison_inputs_to_reactives", "Portfolio_Comparison_Inputs"),
            (
                "save_portfolio_comparison_outputs_to_reactives",
                "Portfolio_Comparison_Outputs",
            ),
            ("save_weights_analysis_inputs_to_reactives", "Weights_Analysis_Inputs"),
            ("save_weights_analysis_outputs_to_reactives", "Weights_Analysis_Outputs"),
            (
                "save_portfolio_optimization_skfolio_inputs_to_reactives",
                "Portfolio_Optimization_Skfolio_Inputs",
            ),
            (
                "save_portfolio_optimization_skfolio_outputs_to_reactives",
                "Portfolio_Optimization_Skfolio_Outputs",
            ),
            (
                "save_portfolio_simulation_inputs_to_reactives",
                "Portfolio_Simulation_Inputs",
            ),
            (
                "save_portfolio_simulation_outputs_to_reactives",
                "Portfolio_Simulation_Outputs",
            ),
        ],
    )
    def test_wrapper_writes_to_correct_key(
        self,
        reactives_with_data_clients: dict[str, Any],
        wrapper_name: str,
        subtab_key: str,
    ) -> None:
        """Wrapper saves a sentinel dict then get_results_data retrieves it."""
        import importlib

        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import get_results_data_from_reactives

        module = importlib.import_module("src.dashboard.shiny_utils.utils_reporting")
        wrapper = getattr(module, wrapper_name)
        sentinel = {"wrapper_test": wrapper_name}

        with reactive.isolate():
            wrapper(reactives_with_data_clients, sentinel)
            result = get_results_data_from_reactives(
                reactives_with_data_clients, subtab_key,
            )
        assert result == sentinel


# =============================================================================
# _coerce_results_value_to_json
# =============================================================================


@pytest.mark.unit()
class TestCoerceResultsValueToJson:
    """``_coerce_results_value_to_json`` converts raw values to serialisable dicts."""

    def test_none_returns_empty_dict(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import _coerce_results_value_to_json

        assert _coerce_results_value_to_json(None) == {}

    def test_dict_returned_as_is(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import _coerce_results_value_to_json

        payload = {"a": 1, "b": "hello"}
        assert _coerce_results_value_to_json(payload) is payload

    def test_list_wrapped_in_records(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import _coerce_results_value_to_json

        result = _coerce_results_value_to_json([1, 2, 3])
        assert "records" in result
        assert result["records"] == [1, 2, 3]

    def test_polars_dataframe_wrapped_in_records(self) -> None:
        import polars as pl

        from src.dashboard.shiny_utils.utils_reporting import _coerce_results_value_to_json

        df = pl.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        result = _coerce_results_value_to_json(df)
        assert "records" in result
        assert isinstance(result["records"], list)
        assert len(result["records"]) == 2

    def test_arbitrary_type_produces_value_str(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import _coerce_results_value_to_json

        result = _coerce_results_value_to_json(42)
        assert "value" in result
        assert result["value"] == "42"


# =============================================================================
# build_results_data_json_from_reactives
# =============================================================================


@pytest.mark.unit()
class TestBuildResultsDataJsonFromReactives:
    """``build_results_data_json_from_reactives`` assembles JSON-ready dicts."""

    def test_returns_empty_dict_when_no_data_stored(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_results_data_json_from_reactives,
            save_results_data_to_reactives,
        )

        # Ensure the key slot is cleared first
        with reactive.isolate():
            save_results_data_to_reactives(
                reactives_with_data_clients, "Portfolio_Comparison_Inputs", None,
            )
            result = build_results_data_json_from_reactives(
                reactives_with_data_clients, "Portfolio_Comparison_Inputs",
            )
        assert result == {}

    def test_returns_dict_payload_when_dict_stored(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_results_data_json_from_reactives,
            save_results_data_to_reactives,
        )

        payload = {"time_period": "3Y", "metric": "sharpe"}
        with reactive.isolate():
            save_results_data_to_reactives(
                reactives_with_data_clients,
                "Portfolio_Analysis_Inputs",
                payload,
            )
            result = build_results_data_json_from_reactives(
                reactives_with_data_clients, "Portfolio_Analysis_Inputs",
            )
        assert result == payload

    def test_invalid_subtab_key_raises_value_error(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_results_data_json_from_reactives,
        )

        with pytest.raises(ValueError), reactive.isolate():
            build_results_data_json_from_reactives(
                reactives_with_data_clients, "Unknown_Key",
            )

    def test_polars_df_stored_produces_records_key(
        self,
        reactives_with_data_clients: dict[str, Any],
        sample_assets_df: Any,
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_results_data_json_from_reactives,
            save_results_data_to_reactives,
        )

        with reactive.isolate():
            save_results_data_to_reactives(
                reactives_with_data_clients,
                "Weights_Analysis_Inputs",
                sample_assets_df,
            )
            result = build_results_data_json_from_reactives(
                reactives_with_data_clients, "Weights_Analysis_Inputs",
            )
        assert "records" in result
        assert isinstance(result["records"], list)

    @pytest.mark.parametrize(
        "subtab_key",
        [
            "Portfolio_Analysis_Inputs",
            "Portfolio_Analysis_Outputs",
            "Portfolio_Comparison_Inputs",
            "Portfolio_Comparison_Outputs",
            "Weights_Analysis_Inputs",
            "Weights_Analysis_Outputs",
            "Portfolio_Optimization_Skfolio_Inputs",
            "Portfolio_Optimization_Skfolio_Outputs",
            "Portfolio_Simulation_Inputs",
            "Portfolio_Simulation_Outputs",
        ],
    )
    def test_all_valid_subtab_keys_return_dict(
        self,
        reactives_with_data_clients: dict[str, Any],
        subtab_key: str,
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.utils_reporting import (
            build_results_data_json_from_reactives,
        )

        with reactive.isolate():
            result = build_results_data_json_from_reactives(
                reactives_with_data_clients, subtab_key,
            )
        assert isinstance(result, dict)


# =============================================================================
# VALID_RESULTS_SUBTAB_KEYS — constant regression
# =============================================================================


@pytest.mark.regression()
class TestValidResultsSubtabKeys:
    """``VALID_RESULTS_SUBTAB_KEYS`` must match ``Data_Results`` as initialised."""

    def test_contains_ten_keys(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import VALID_RESULTS_SUBTAB_KEYS

        assert len(VALID_RESULTS_SUBTAB_KEYS) == 10

    def test_all_keys_present_in_data_results(
        self, reactives_with_data_clients: dict[str, Any],
    ) -> None:
        from src.dashboard.shiny_utils.utils_reporting import VALID_RESULTS_SUBTAB_KEYS

        data_results = reactives_with_data_clients["Data_Results"]
        missing = [k for k in VALID_RESULTS_SUBTAB_KEYS if k not in data_results]
        assert missing == [], f"Keys in VALID_RESULTS_SUBTAB_KEYS missing from Data_Results: {missing}"

    def test_is_frozenset(self) -> None:
        from src.dashboard.shiny_utils.utils_reporting import VALID_RESULTS_SUBTAB_KEYS

        assert isinstance(VALID_RESULTS_SUBTAB_KEYS, frozenset)
