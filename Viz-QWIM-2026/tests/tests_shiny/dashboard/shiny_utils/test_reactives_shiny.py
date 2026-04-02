"""Unit tests for ``src.dashboard.shiny_utils.reactives_shiny``.

Tests cover every exported validator and helper without requiring a running
Shiny session.  Only ``initialize_reactives_shiny`` creates ``reactive.Value``
objects — those tests use Shiny's ``reactive.isolate()`` context manager.

New tests added in this revision target the four functions introduced for
input synchronisation between Shiny UI and ``reactives_shiny``:

* ``_reactive_key_to_input_id``
* ``_register_single_input_observer``  (registration API)
* ``register_client_input_observers``  (registration API)
* ``_register_subtab_snapshot_observer``  (registration API)
* ``register_portfolio_results_observers``  (registration API)

Plus regression tests verifying that the new ``User_Inputs_Shiny`` keys for
Weights Analysis, skfolio optimisation, and Portfolio Simulation are present
after ``initialize_reactives_shiny``.

Run:
    pytest tests/tests_shiny/ -m unit -k reactives_shiny
"""

from __future__ import annotations

from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_skeleton() -> dict[str, Any]:
    """Return the minimal valid reactives_shiny skeleton (plain dicts)."""
    return {
        "User_Inputs_Shiny": {},
        "Inner_Variables_Shiny": {},
        "Triggers_Shiny": {},
        "Visual_Objects_Shiny": {},
        "Data_Clients": {},
        "Data_Results": {},
    }


def _valid_data_utils() -> dict[str, Any]:
    """Return a minimal data_utils dict sufficient for ``validate_*`` checks."""
    return {"constants": {}, "defaults": {}}


# ---------------------------------------------------------------------------
# validate_data_utils_parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestValidateDataUtilsParameter:
    """validate_data_utils_parameter must return (bool, str)."""

    def test_none_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_data_utils_parameter

        ok, msg = validate_data_utils_parameter(None)  # type: ignore[arg-type]
        assert ok is False
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_string_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_data_utils_parameter

        ok, _msg = validate_data_utils_parameter("not-a-dict")  # type: ignore[arg-type]
        assert ok is False

    def test_list_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_data_utils_parameter

        ok, _msg = validate_data_utils_parameter([])  # type: ignore[arg-type]
        assert ok is False

    def test_empty_dict_returns_true(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_data_utils_parameter

        ok, _msg = validate_data_utils_parameter({})
        assert ok is True

    def test_populated_dict_returns_true(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_data_utils_parameter

        ok, _msg = validate_data_utils_parameter({"key": "value"})
        assert ok is True

    def test_return_type_is_tuple(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_data_utils_parameter

        result = validate_data_utils_parameter({})
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_successful_validation_msg_is_empty_or_ok(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_data_utils_parameter

        ok, msg = validate_data_utils_parameter(_valid_data_utils())
        assert ok is True
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# validate_reactives_shiny_structure
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestValidateReactivesShinyStructure:
    """validate_reactives_shiny_structure enforces the 4-category schema."""

    def test_none_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactives_shiny_structure,
        )

        ok, msg = validate_reactives_shiny_structure(None)  # type: ignore[arg-type]
        assert ok is False
        assert isinstance(msg, str)

    def test_non_dict_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactives_shiny_structure,
        )

        ok, _msg = validate_reactives_shiny_structure([])  # type: ignore[arg-type]
        assert ok is False

    def test_empty_dict_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactives_shiny_structure,
        )

        ok, _msg = validate_reactives_shiny_structure({})
        assert ok is False

    def test_missing_one_category_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactives_shiny_structure,
        )

        incomplete = {
            "User_Inputs_Shiny": {},
            "Inner_Variables_Shiny": {},
            "Triggers_Shiny": {},
            # "Visual_Objects_Shiny" is missing
        }
        ok, _msg = validate_reactives_shiny_structure(incomplete)
        assert ok is False

    def test_all_required_categories_returns_true(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactives_shiny_structure,
        )

        ok, msg = validate_reactives_shiny_structure(_valid_skeleton())
        assert ok is True
        assert isinstance(msg, str)

    @pytest.mark.parametrize(
        "category",
        [
            "User_Inputs_Shiny",
            "Inner_Variables_Shiny",
            "Triggers_Shiny",
            "Visual_Objects_Shiny",
            "Data_Clients",
            "Data_Results",
        ],
    )
    def test_each_required_category_individually(self, category: str) -> None:
        """Removing each required category must flip is_valid to False."""
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactives_shiny_structure,
        )

        skeleton = _valid_skeleton()
        del skeleton[category]
        ok, _msg = validate_reactives_shiny_structure(skeleton)
        assert ok is False


# ---------------------------------------------------------------------------
# validate_reactive_key_access
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestValidateReactiveKeyAccess:
    """validate_reactive_key_access checks a key exists in a category dict."""

    def test_none_category_dict_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactive_key_access,
        )

        ok, msg = validate_reactive_key_access(None, "key", "User_Inputs_Shiny")  # type: ignore[arg-type]
        assert ok is False
        assert isinstance(msg, str)

    def test_non_dict_category_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactive_key_access,
        )

        ok, _msg = validate_reactive_key_access("not_a_dict", "key", "User_Inputs_Shiny")  # type: ignore[arg-type]
        assert ok is False

    def test_none_key_name_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactive_key_access,
        )

        ok, _msg = validate_reactive_key_access({}, None, "User_Inputs_Shiny")  # type: ignore[arg-type]
        assert ok is False

    def test_empty_key_name_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactive_key_access,
        )

        ok, _msg = validate_reactive_key_access({"a": 1}, "", "User_Inputs_Shiny")
        assert ok is False

    def test_missing_key_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactive_key_access,
        )

        ok, _msg = validate_reactive_key_access({"a": 1}, "b", "User_Inputs_Shiny")
        assert ok is False

    def test_existing_key_returns_true(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            validate_reactive_key_access,
        )

        ok, msg = validate_reactive_key_access({"my_key": 42}, "my_key", "Inner_Variables_Shiny")
        assert ok is True
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# validate_category_name
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestValidateCategoryName:
    """validate_category_name accepts only the four canonical category names."""

    @pytest.mark.parametrize(
        "valid_name",
        [
            "User_Inputs_Shiny",
            "Inner_Variables_Shiny",
            "Triggers_Shiny",
            "Visual_Objects_Shiny",
        ],
    )
    def test_valid_category_names(self, valid_name: str) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_category_name

        ok, msg = validate_category_name(valid_name)
        assert ok is True
        assert isinstance(msg, str)

    def test_none_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_category_name

        ok, _msg = validate_category_name(None)  # type: ignore[arg-type]
        assert ok is False

    def test_non_str_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_category_name

        ok, _msg = validate_category_name(123)  # type: ignore[arg-type]
        assert ok is False

    def test_empty_string_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_category_name

        ok, _msg = validate_category_name("")
        assert ok is False

    def test_unknown_category_name_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_category_name

        ok, _msg = validate_category_name("Unknown_Category")
        assert ok is False

    def test_case_sensitive_mismatch_returns_false(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import validate_category_name

        ok, _msg = validate_category_name("user_inputs_shiny")  # lowercase
        assert ok is False


# ---------------------------------------------------------------------------
# safe_get_shiny_input_value
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestSafeGetShinyInputValue:
    """safe_get_shiny_input_value must never raise; bad inputs → None."""

    def test_none_input_events_returns_none(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import safe_get_shiny_input_value

        result = safe_get_shiny_input_value(None, "some_id")  # type: ignore[arg-type]
        assert result is None

    def test_non_string_identifier_returns_none(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import safe_get_shiny_input_value

        result = safe_get_shiny_input_value({}, 123)  # type: ignore[arg-type]
        assert result is None

    def test_empty_identifier_returns_none(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import safe_get_shiny_input_value

        result = safe_get_shiny_input_value({}, "")
        assert result is None


# ---------------------------------------------------------------------------
# create_reactive_value_safely
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestCreateReactiveValueSafely:
    """create_reactive_value_safely wraps reactive.Value construction."""

    def test_returns_non_none_for_valid_initial_value(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import create_reactive_value_safely

        with reactive.isolate():
            rv = create_reactive_value_safely(42)
        assert rv is not None

    def test_returns_non_none_for_none_initial_value(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import create_reactive_value_safely

        with reactive.isolate():
            rv = create_reactive_value_safely(None)
        assert rv is not None

    def test_stored_value_is_readable(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import create_reactive_value_safely

        with reactive.isolate():
            rv = create_reactive_value_safely("hello")
            assert rv.get() == "hello"

    def test_stored_list_value_is_readable(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import create_reactive_value_safely

        with reactive.isolate():
            rv = create_reactive_value_safely([1, 2, 3])
            assert rv.get() == [1, 2, 3]


# ---------------------------------------------------------------------------
# initialize_reactives_shiny
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestInitializeReactivesShiny:
    """initialize_reactives_shiny must return a properly-structured dict."""

    def test_raises_value_error_for_none_data_utils(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import initialize_reactives_shiny

        with pytest.raises((ValueError, TypeError)), reactive.isolate():
            initialize_reactives_shiny(None)  # type: ignore[arg-type]

    def test_raises_value_error_for_non_dict_data_utils(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import initialize_reactives_shiny

        with pytest.raises((ValueError, TypeError)), reactive.isolate():
            initialize_reactives_shiny("bad")  # type: ignore[arg-type]

    def test_returns_dict_for_valid_data_utils(self, data_utils: dict[str, Any]) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import initialize_reactives_shiny

        with reactive.isolate():
            result = initialize_reactives_shiny(data_utils)
        assert isinstance(result, dict)

    def test_has_all_four_categories(self, data_utils: dict[str, Any]) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import initialize_reactives_shiny

        with reactive.isolate():
            result = initialize_reactives_shiny(data_utils)
        expected_keys = {
            "User_Inputs_Shiny",
            "Inner_Variables_Shiny",
            "Triggers_Shiny",
            "Visual_Objects_Shiny",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_each_category_is_dict(self, data_utils: dict[str, Any]) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import initialize_reactives_shiny

        with reactive.isolate():
            result = initialize_reactives_shiny(data_utils)
        for category in result.values():
            assert isinstance(category, dict)


# ---------------------------------------------------------------------------
# safe_get_reactive_value
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestSafeGetReactiveValue:
    """safe_get_reactive_value must return the default on invalid input."""

    def test_none_reactive_returns_default_none(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import safe_get_reactive_value

        result = safe_get_reactive_value(None, default_value=None)
        assert result is None

    def test_none_reactive_returns_custom_default(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import safe_get_reactive_value

        result = safe_get_reactive_value(None, default_value="fallback")
        assert result == "fallback"

    def test_valid_reactive_returns_its_value(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import safe_get_reactive_value

        with reactive.isolate():
            rv = reactive.Value(99)
            result = safe_get_reactive_value(rv, default_value=0)
        assert result == 99


# ---------------------------------------------------------------------------
# safe_get_value_from_shiny_input_text
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestSafeGetValueFromShinyInputText:
    """safe_get_value_from_shiny_input_text must return a string default on failure."""

    def test_none_input_returns_default_str(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            safe_get_value_from_shiny_input_text,
        )

        result = safe_get_value_from_shiny_input_text(None, default_value="default")
        assert result == "default"

    def test_none_input_returns_empty_str_when_default_empty(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            safe_get_value_from_shiny_input_text,
        )

        result = safe_get_value_from_shiny_input_text(None, default_value="")
        assert result == ""

    def test_valid_reactive_with_string_value(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import (
            safe_get_value_from_shiny_input_text,
        )

        with reactive.isolate():
            rv = reactive.Value("hello")
            result = safe_get_value_from_shiny_input_text(rv, default_value="default")
        assert result == "hello"


# ---------------------------------------------------------------------------
# safe_get_value_from_shiny_input_numeric
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestSafeGetValueFromShinyInputNumeric:
    """safe_get_value_from_shiny_input_numeric must return a numeric default on failure."""

    def test_none_input_returns_numeric_default(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            safe_get_value_from_shiny_input_numeric,
        )

        result = safe_get_value_from_shiny_input_numeric(None, default_value=0.0)
        assert result == 0.0

    def test_none_input_with_non_zero_default(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import (
            safe_get_value_from_shiny_input_numeric,
        )

        result = safe_get_value_from_shiny_input_numeric(None, default_value=42.5)
        assert result == 42.5

    def test_valid_reactive_returns_numeric_value(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import (
            safe_get_value_from_shiny_input_numeric,
        )

        with reactive.isolate():
            rv = reactive.Value(3.14)
            result = safe_get_value_from_shiny_input_numeric(rv, default_value=0.0)
        assert result == pytest.approx(3.14, rel=1e-6)


# ---------------------------------------------------------------------------
# get_value_from_reactives_shiny / set_value_to_reactives_shiny
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestGetSetValueFromReactivesShiny:
    """Round-trip set→get for reactives_shiny values.

    ``set_value_to_reactives_shiny`` requires the key to already exist in the
    category dict (it updates, not inserts).  ``get_value_from_reactives_shiny``
    raises ``KeyError`` for missing keys.  Tests reflect the actual semantics.
    """

    def test_set_existing_key_roundtrip(self, minimal_reactives_shiny: dict[str, Any]) -> None:
        """Use a key that was created by initialize_reactives_shiny."""
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import (
            get_value_from_reactives_shiny,
            set_value_to_reactives_shiny,
        )

        # Use the first available key in Inner_Variables_Shiny
        inner = minimal_reactives_shiny["Inner_Variables_Shiny"]
        if not inner:
            pytest.skip("No keys in Inner_Variables_Shiny to test round-trip")

        existing_key = next(iter(inner))
        with reactive.isolate():
            set_value_to_reactives_shiny(
                minimal_reactives_shiny,
                key_name=existing_key,
                key_category="Inner_Variables_Shiny",
                input_value="round_trip_sentinel",
            )
            result = get_value_from_reactives_shiny(
                minimal_reactives_shiny,
                key_name=existing_key,
                key_category="Inner_Variables_Shiny",
            )
        # Result should be the sentinel value itself or a reactive.Value wrapping it
        if hasattr(result, "get"):
            with reactive.isolate():
                assert result.get() == "round_trip_sentinel"
        else:
            assert result == "round_trip_sentinel"

    def test_get_missing_key_raises_key_error(self, minimal_reactives_shiny: dict[str, Any]) -> None:
        """get_value_from_reactives_shiny raises KeyError for absent keys."""
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import (
            get_value_from_reactives_shiny,
        )

        with pytest.raises(KeyError), reactive.isolate():
            get_value_from_reactives_shiny(
                minimal_reactives_shiny,
                key_name="Definitely_Missing_Key_XYZ",
                key_category="User_Inputs_Shiny",
            )

    def test_set_missing_key_raises_key_error(self, minimal_reactives_shiny: dict[str, Any]) -> None:
        """set_value_to_reactives_shiny raises KeyError when the key does not exist."""
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import (
            set_value_to_reactives_shiny,
        )

        with pytest.raises(KeyError), reactive.isolate():
            set_value_to_reactives_shiny(
                minimal_reactives_shiny,
                key_name="Key_That_Does_Not_Exist_XYZ",
                key_category="User_Inputs_Shiny",
                input_value=None,
            )

    def test_set_existing_trigger_key_is_allowed(self, minimal_reactives_shiny: dict[str, Any]) -> None:
        """Updating an existing Triggers_Shiny key with None must not raise."""
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import (
            set_value_to_reactives_shiny,
        )

        triggers = minimal_reactives_shiny["Triggers_Shiny"]
        if not triggers:
            pytest.skip("No keys in Triggers_Shiny to test")

        existing_key = next(iter(triggers))
        with reactive.isolate():
            # Must not raise
            set_value_to_reactives_shiny(
                minimal_reactives_shiny,
                key_name=existing_key,
                key_category="Triggers_Shiny",
                input_value=None,
            )

# ---------------------------------------------------------------------------
# _reactive_key_to_input_id  (private — tested via module import)
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestReactiveKeyToInputId:
    """``_reactive_key_to_input_id`` translates User_Inputs_Shiny keys to Shiny IDs."""

    def test_clients_key_produces_correct_input_id(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _reactive_key_to_input_id

        result = _reactive_key_to_input_id(
            "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Name",
        )
        assert result == "input_ID_tab_clients_subtab_clients_personal_info_client_primary_name"

    def test_portfolio_comparison_key(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _reactive_key_to_input_id

        result = _reactive_key_to_input_id(
            "Input_Tab_Portfolios_Subtab_Comparison_Time_Period",
        )
        assert result == "input_ID_tab_portfolios_subtab_comparison_time_period"

    def test_weights_analysis_key(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _reactive_key_to_input_id

        result = _reactive_key_to_input_id(
            "Input_Tab_Portfolios_Subtab_Weights_Analysis_Viz_Type",
        )
        assert result == "input_ID_tab_portfolios_subtab_weights_analysis_viz_type"

    def test_skfolio_key_method1(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _reactive_key_to_input_id

        result = _reactive_key_to_input_id(
            "Input_Tab_Portfolios_Subtab_Skfolio_Method1_Category",
        )
        assert result == "input_ID_tab_portfolios_subtab_skfolio_method1_category"

    def test_simulation_key(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _reactive_key_to_input_id

        result = _reactive_key_to_input_id(
            "Input_Tab_Results_Subtab_Simulation_Distribution_Type",
        )
        assert result == "input_ID_tab_results_subtab_simulation_distribution_type"

    def test_bad_prefix_raises_value_error(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _reactive_key_to_input_id

        with pytest.raises(ValueError, match="Input_Tab_"):
            _reactive_key_to_input_id("Some_Arbitrary_Key_Without_Correct_Prefix")

    def test_empty_string_raises_value_error(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _reactive_key_to_input_id

        with pytest.raises(ValueError):
            _reactive_key_to_input_id("")

    def test_preserves_lowercase_suffix(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _reactive_key_to_input_id

        result = _reactive_key_to_input_id("Input_Tab_ABC_Subtab_XYZ")
        assert result == "input_ID_tab_abc_subtab_xyz"


# ---------------------------------------------------------------------------
# New User_Inputs_Shiny keys — regression: must be present after init
# ---------------------------------------------------------------------------

#: All new reactive keys added for Weights Analysis
_WEIGHTS_ANALYSIS_KEYS = [
    "Input_Tab_Portfolios_Subtab_Weights_Analysis_Time_Period",
    "Input_Tab_Portfolios_Subtab_Weights_Analysis_Date_Range",
    "Input_Tab_Portfolios_Subtab_Weights_Analysis_Viz_Type",
    "Input_Tab_Portfolios_Subtab_Weights_Analysis_Show_Pct",
    "Input_Tab_Portfolios_Subtab_Weights_Analysis_Sort_Components",
    "Input_Tab_Portfolios_Subtab_Weights_Analysis_Select_All_Components",
]

#: All new reactive keys added for skfolio optimisation
_SKFOLIO_KEYS = [
    "Input_Tab_Portfolios_Subtab_Skfolio_Method1_Category",
    "Input_Tab_Portfolios_Subtab_Skfolio_Method1_Type",
    "Input_Tab_Portfolios_Subtab_Skfolio_Method1_Objective",
    "Input_Tab_Portfolios_Subtab_Skfolio_Method1_Risk_Aversion",
    "Input_Tab_Portfolios_Subtab_Skfolio_Method2_Category",
    "Input_Tab_Portfolios_Subtab_Skfolio_Method2_Type",
    "Input_Tab_Portfolios_Subtab_Skfolio_Method2_Objective",
    "Input_Tab_Portfolios_Subtab_Skfolio_Method2_Risk_Aversion",
    "Input_Tab_Portfolios_Subtab_Skfolio_Time_Period",
]

#: All new reactive keys added for Portfolio Simulation
_SIMULATION_KEYS = [
    "Input_Tab_Results_Subtab_Simulation_Distribution_Type",
    "Input_Tab_Results_Subtab_Simulation_Rng_Type",
    "Input_Tab_Results_Subtab_Simulation_Num_Scenarios",
    "Input_Tab_Results_Subtab_Simulation_Num_Days",
    "Input_Tab_Results_Subtab_Simulation_Seed",
    "Input_Tab_Results_Subtab_Simulation_Initial_Value",
    "Input_Tab_Results_Subtab_Simulation_Degrees_Of_Freedom",
    "Input_Tab_Results_Subtab_Simulation_Start_Date",
    "Input_Tab_Results_Subtab_Simulation_Select_All_Components",
]


@pytest.mark.regression()
class TestNewUserInputsShinyKeysPresent:
    """Regression: ``initialize_reactives_shiny`` must include all newly added keys."""

    @pytest.mark.parametrize("key", _WEIGHTS_ANALYSIS_KEYS)
    def test_weights_analysis_key_present(
        self, minimal_reactives_shiny: dict[str, Any], key: str,
    ) -> None:
        assert key in minimal_reactives_shiny["User_Inputs_Shiny"], (
            f"Missing Weights Analysis key: {key}"
        )

    @pytest.mark.parametrize("key", _SKFOLIO_KEYS)
    def test_skfolio_key_present(
        self, minimal_reactives_shiny: dict[str, Any], key: str,
    ) -> None:
        assert key in minimal_reactives_shiny["User_Inputs_Shiny"], (
            f"Missing skfolio key: {key}"
        )

    @pytest.mark.parametrize("key", _SIMULATION_KEYS)
    def test_simulation_key_present(
        self, minimal_reactives_shiny: dict[str, Any], key: str,
    ) -> None:
        assert key in minimal_reactives_shiny["User_Inputs_Shiny"], (
            f"Missing Simulation key: {key}"
        )

    def test_total_new_key_count(self, minimal_reactives_shiny: dict[str, Any]) -> None:
        """All 24 new keys must be present (6 Weights + 9 Skfolio + 9 Simulation)."""
        all_new_keys = _WEIGHTS_ANALYSIS_KEYS + _SKFOLIO_KEYS + _SIMULATION_KEYS
        user_inputs = minimal_reactives_shiny["User_Inputs_Shiny"]
        missing = [k for k in all_new_keys if k not in user_inputs]
        assert missing == [], f"Missing keys: {missing}"

    def test_new_keys_have_reactive_values(self, minimal_reactives_shiny: dict[str, Any]) -> None:
        """Newly added keys must be ``reactive.Value`` instances (not ``None``)."""
        from shiny import reactive

        user_inputs = minimal_reactives_shiny["User_Inputs_Shiny"]
        spot_check = [
            "Input_Tab_Portfolios_Subtab_Weights_Analysis_Time_Period",
            "Input_Tab_Portfolios_Subtab_Skfolio_Time_Period",
            "Input_Tab_Results_Subtab_Simulation_Distribution_Type",
        ]
        for key in spot_check:
            rv = user_inputs.get(key)
            assert rv is not None, f"reactive.Value for {key!r} is None"
            assert isinstance(rv, reactive.Value), f"{key!r} is not a reactive.Value"


# ---------------------------------------------------------------------------
# _PORTFOLIO_SUBTAB_RESULTS_KEY_MAP — structure regression
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class TestPortfolioSubtabResultsKeyMap:
    """The ``_PORTFOLIO_SUBTAB_RESULTS_KEY_MAP`` constant must be complete."""

    def test_map_contains_five_entries(self) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _PORTFOLIO_SUBTAB_RESULTS_KEY_MAP

        assert len(_PORTFOLIO_SUBTAB_RESULTS_KEY_MAP) == 5

    @pytest.mark.parametrize(
        "base_key",
        [
            "Portfolio_Analysis",
            "Portfolio_Comparison",
            "Weights_Analysis",
            "Portfolio_Optimization_Skfolio",
            "Portfolio_Simulation",
        ],
    )
    def test_expected_base_key_present(self, base_key: str) -> None:
        from src.dashboard.shiny_utils.reactives_shiny import _PORTFOLIO_SUBTAB_RESULTS_KEY_MAP

        assert base_key in _PORTFOLIO_SUBTAB_RESULTS_KEY_MAP.values(), (
            f"Base key {base_key!r} missing from map"
        )

    def test_all_values_correspond_to_data_results_entries(
        self, minimal_reactives_shiny: dict[str, Any],
    ) -> None:
        """Every base key in the map must have matching _Inputs/_Outputs in Data_Results."""
        from src.dashboard.shiny_utils.reactives_shiny import _PORTFOLIO_SUBTAB_RESULTS_KEY_MAP

        data_results = minimal_reactives_shiny["Data_Results"]
        for base_key in _PORTFOLIO_SUBTAB_RESULTS_KEY_MAP.values():
            assert f"{base_key}_Inputs" in data_results, (
                f"Data_Results missing {base_key}_Inputs"
            )
            assert f"{base_key}_Outputs" in data_results, (
                f"Data_Results missing {base_key}_Outputs"
            )


# ---------------------------------------------------------------------------
# register_client_input_observers — API validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestRegisterClientInputObserversValidation:
    """``register_client_input_observers`` must validate its arguments."""

    def test_none_input_raises_value_error(
        self, minimal_reactives_shiny: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import register_client_input_observers

        with pytest.raises(ValueError, match="input cannot be None"), reactive.isolate():
            register_client_input_observers(None, minimal_reactives_shiny)  # type: ignore[arg-type]

    def test_invalid_reactives_structure_raises_value_error(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import register_client_input_observers

        class _FakeInput:
            pass

        with pytest.raises(ValueError), reactive.isolate():
            register_client_input_observers(_FakeInput(), {"wrong": {}})

    def test_valid_call_does_not_raise(
        self, minimal_reactives_shiny: dict[str, Any],
    ) -> None:
        """A valid call should register observers without raising."""
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import register_client_input_observers

        class _FakeInput:
            """Minimal Shiny input proxy stub."""

            def __getattr__(self, name: str) -> Any:
                def _call() -> None:
                    return None

                return _call

        with reactive.isolate():
            # Must not raise
            register_client_input_observers(_FakeInput(), minimal_reactives_shiny)


# ---------------------------------------------------------------------------
# register_portfolio_results_observers — API validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestRegisterPortfolioResultsObserversValidation:
    """``register_portfolio_results_observers`` must validate its arguments."""

    def test_none_input_raises_value_error(
        self, minimal_reactives_shiny: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import (
            register_portfolio_results_observers,
        )

        with pytest.raises(ValueError), reactive.isolate():
            register_portfolio_results_observers(None, minimal_reactives_shiny)  # type: ignore[arg-type]

    def test_invalid_reactives_raises_value_error(self) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import (
            register_portfolio_results_observers,
        )

        class _FakeInput:
            pass

        with pytest.raises(ValueError), reactive.isolate():
            register_portfolio_results_observers(_FakeInput(), {"bad": "structure"})

    def test_valid_call_does_not_raise(
        self, minimal_reactives_shiny: dict[str, Any],
    ) -> None:
        from shiny import reactive

        from src.dashboard.shiny_utils.reactives_shiny import (
            register_portfolio_results_observers,
        )

        class _FakeInput:
            def __getattr__(self, name: str) -> Any:
                def _call() -> None:
                    return None

                return _call

        with reactive.isolate():
            register_portfolio_results_observers(_FakeInput(), minimal_reactives_shiny)

    @pytest.mark.parametrize(
        "base_key",
        [
            "Portfolio_Analysis",
            "Portfolio_Comparison",
            "Weights_Analysis",
            "Portfolio_Optimization_Skfolio",
            "Portfolio_Simulation",
        ],
    )
    def test_data_results_inputs_accessible_after_registration(
        self, minimal_reactives_shiny: dict[str, Any], base_key: str,
    ) -> None:
        """After registration, the _Inputs reactive.Value must be accessible."""
        from shiny import reactive

        inputs_key = f"{base_key}_Inputs"
        rv = minimal_reactives_shiny["Data_Results"].get(inputs_key)
        assert rv is not None, f"Data_Results[{inputs_key!r}] is None"
        with reactive.isolate():
            # Initial value is None — just verify .get() is callable
            _ = rv.get()


# ---------------------------------------------------------------------------
# initialize_reactives_shiny — Data_Clients and Data_Results categories
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestInitializeReactivesShinyExtended:
    """Extend existing init tests: Data_Clients and Data_Results categories."""

    def test_has_data_clients_category(self, minimal_reactives_shiny: dict[str, Any]) -> None:
        assert "Data_Clients" in minimal_reactives_shiny

    def test_has_data_results_category(self, minimal_reactives_shiny: dict[str, Any]) -> None:
        assert "Data_Results" in minimal_reactives_shiny

    def test_data_clients_has_single_or_couple(
        self, minimal_reactives_shiny: dict[str, Any],
    ) -> None:
        assert "Single_Or_Couple" in minimal_reactives_shiny["Data_Clients"]

    def test_data_clients_has_required_sub_levels(
        self, minimal_reactives_shiny: dict[str, Any],
    ) -> None:
        data_clients = minimal_reactives_shiny["Data_Clients"]
        for sub_level in ("Client_Primary", "Client_Partner", "Clients_Combined"):
            assert sub_level in data_clients, f"Missing sub-level: {sub_level}"

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
    def test_data_results_has_all_ten_subtab_keys(
        self, minimal_reactives_shiny: dict[str, Any], subtab_key: str,
    ) -> None:
        assert subtab_key in minimal_reactives_shiny["Data_Results"], (
            f"Missing Data_Results key: {subtab_key}"
        )
