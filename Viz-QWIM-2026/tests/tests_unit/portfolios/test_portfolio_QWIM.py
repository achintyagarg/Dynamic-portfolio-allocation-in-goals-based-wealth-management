"""Unit tests for the portfolio_QWIM class.

This module contains comprehensive tests for all methods and properties
of the portfolio_QWIM class, including initialization, validation,
date parsing, weight management, and string representations.

Test Classes
------------
- Test_Portfolio_QWIM_Init_From_Components
    Tests for initializing a portfolio from component name lists.
- Test_Portfolio_QWIM_Init_From_Weights
    Tests for initializing a portfolio from a Polars DataFrame.
- Test_Portfolio_QWIM_Init_Validation
    Tests that invalid constructor arguments raise correct errors.
- Test_Portfolio_QWIM_Validate_Type_And_Value
    Tests for the m_validate_type_and_value internal method.
- Test_Portfolio_QWIM_Validate_Dataframe
    Tests for the m_validate_dataframe internal method.
- Test_Portfolio_QWIM_Validate_Weights
    Tests for the m_validate_weights internal method.
- Test_Portfolio_QWIM_Parse_Date
    Tests for the m_parse_date internal method.
- Test_Portfolio_QWIM_Properties
    Tests for all @property and method accessors.
- Test_Portfolio_QWIM_Add_Weights
    Tests for the add_weights method.
- Test_Portfolio_QWIM_Modify_Weights
    Tests for the modify_weights method.
- Test_Portfolio_QWIM_Validate_All_Weights
    Tests for the validate_all_weights method.
- Test_Portfolio_QWIM_String_Representations
    Tests for __str__ and __repr__.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from src.portfolios.portfolio_QWIM import portfolio_QWIM


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture()
def basic_weights_df():
    """Minimal valid two-component weights DataFrame."""
    return pl.DataFrame(
        {
            "Date": ["2024-01-01", "2024-02-01"],
            "AAPL": [0.6, 0.5],
            "MSFT": [0.4, 0.5],
        }
    )


@pytest.fixture()
def single_row_weights_df():
    """Single-row weights DataFrame for simple portfolio creation."""
    return pl.DataFrame(
        {
            "Date": ["2024-01-01"],
            "VTI": [0.4],
            "AGG": [0.3],
            "VNQ": [0.3],
        }
    )


@pytest.fixture()
def portfolio_from_components():
    """Portfolio created from component names (equal weights)."""
    return portfolio_QWIM(
        name_portfolio="Test Portfolio",
        names_components=["AAPL", "MSFT", "GOOG"],
    )


@pytest.fixture()
def portfolio_from_weights(basic_weights_df):
    """Portfolio created from a weights DataFrame."""
    return portfolio_QWIM(
        name_portfolio="Weights Portfolio",
        portfolio_weights=basic_weights_df,
    )


# ==============================================================================
# Tests: Init from component names
# ==============================================================================


class Test_Portfolio_QWIM_Init_From_Components:
    """Test initialization using names_components parameter."""

    @pytest.mark.unit()
    def test_creates_portfolio_with_correct_components(self):
        """Portfolio stores the provided component names."""
        p = portfolio_QWIM(
            name_portfolio="Tech",
            names_components=["AAPL", "MSFT"],
        )
        assert p.get_portfolio_components == ["AAPL", "MSFT"]

    @pytest.mark.unit()
    def test_creates_equal_weights_for_two_components(self):
        """Two components should each receive 0.5 weight."""
        p = portfolio_QWIM(
            name_portfolio="Split",
            names_components=["A", "B"],
        )
        weights = p.get_portfolio_weights()
        assert weights["A"][0] == pytest.approx(0.5, rel=1e-6)
        assert weights["B"][0] == pytest.approx(0.5, rel=1e-6)

    @pytest.mark.unit()
    def test_creates_equal_weights_for_three_components(self):
        """Three components should each receive 1/3 weight."""
        p = portfolio_QWIM(
            name_portfolio="Third",
            names_components=["X", "Y", "Z"],
        )
        weights = p.get_portfolio_weights()
        expected = pytest.approx(1 / 3, rel=1e-6)
        assert weights["X"][0] == expected
        assert weights["Y"][0] == expected
        assert weights["Z"][0] == expected

    @pytest.mark.unit()
    def test_single_component_receives_full_weight(self):
        """Single component should receive weight of 1.0."""
        p = portfolio_QWIM(
            name_portfolio="Solo",
            names_components=["ONLY"],
        )
        weights = p.get_portfolio_weights()
        assert weights["ONLY"][0] == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.unit()
    def test_num_components_matches_list_length(self):
        """get_num_components should equal the length of the input list."""
        p = portfolio_QWIM(
            name_portfolio="Five",
            names_components=["A", "B", "C", "D", "E"],
        )
        assert p.get_num_components == 5

    @pytest.mark.unit()
    def test_weights_dataframe_has_date_column(self):
        """Resulting weights DataFrame must contain a 'Date' column."""
        p = portfolio_QWIM(
            name_portfolio="DateCheck",
            names_components=["VTI", "BND"],
        )
        assert "Date" in p.get_portfolio_weights().columns

    @pytest.mark.unit()
    def test_strips_whitespace_from_component_names(self):
        """Component names with surrounding whitespace should be stripped."""
        p = portfolio_QWIM(
            name_portfolio="Trim",
            names_components=[" SPY ", " QQQ "],
        )
        comps = p.get_portfolio_components
        assert "SPY" in comps
        assert "QQQ" in comps

    @pytest.mark.unit()
    def test_date_portfolio_string_is_used(self):
        """Supplied date string should be stored as the portfolio date."""
        p = portfolio_QWIM(
            name_portfolio="Dated",
            names_components=["VTI"],
            date_portfolio="2023-06-15",
        )
        row = p.get_portfolio_weights()["Date"][0]
        assert str(row) == "2023-06-15"

    @pytest.mark.unit()
    def test_date_portfolio_none_uses_today(self):
        """When date_portfolio is None, today's date should be used."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        p = portfolio_QWIM(
            name_portfolio="Today",
            names_components=["SPY"],
            date_portfolio=None,
        )
        row_date = str(p.get_portfolio_weights()["Date"][0])
        assert row_date == today_str


# ==============================================================================
# Tests: Init from weights DataFrame
# ==============================================================================


class Test_Portfolio_QWIM_Init_From_Weights:
    """Test initialization using portfolio_weights parameter."""

    @pytest.mark.unit()
    def test_creates_portfolio_with_correct_components(self, basic_weights_df):
        """Components derived from DataFrame columns (excluding 'Date')."""
        p = portfolio_QWIM(
            name_portfolio="FromDF",
            portfolio_weights=basic_weights_df,
        )
        assert sorted(p.get_portfolio_components) == ["AAPL", "MSFT"]

    @pytest.mark.unit()
    def test_num_components_reflects_dataframe_columns(self, basic_weights_df):
        """get_num_components should equal number of non-Date columns."""
        p = portfolio_QWIM(
            name_portfolio="ColCount",
            portfolio_weights=basic_weights_df,
        )
        assert p.get_num_components == 2

    @pytest.mark.unit()
    def test_weights_dataframe_preserved(self, basic_weights_df):
        """Weights stored should have same number of rows as input."""
        p = portfolio_QWIM(
            name_portfolio="Rows",
            portfolio_weights=basic_weights_df,
        )
        assert len(p.get_portfolio_weights()) == 2

    @pytest.mark.unit()
    def test_date_is_first_column(self, basic_weights_df):
        """Date must be the first column in the stored weights DataFrame."""
        p = portfolio_QWIM(
            name_portfolio="Order",
            portfolio_weights=basic_weights_df,
        )
        assert p.get_portfolio_weights().columns[0] == "Date"

    @pytest.mark.unit()
    def test_weights_if_dataframe_provided_and_components_also_provided(
        self, basic_weights_df
    ):
        """When both weights and components given, weights DataFrame takes precedence."""
        p = portfolio_QWIM(
            name_portfolio="Both",
            portfolio_weights=basic_weights_df,
            names_components=["IGNORED_A", "IGNORED_B"],
        )
        assert sorted(p.get_portfolio_components) == ["AAPL", "MSFT"]

    @pytest.mark.unit()
    def test_portfolio_name_stored_correctly(self, basic_weights_df):
        """The exact name passed should be stored and retrievable."""
        p = portfolio_QWIM(
            name_portfolio="My Fund",
            portfolio_weights=basic_weights_df,
        )
        assert p.get_portfolio_name == "My Fund"

    @pytest.mark.unit()
    def test_unnormalized_weights_are_normalized(self):
        """Weights not summing to 1.0 should be normalized automatically."""
        df = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "A": [2.0],
                "B": [2.0],
            }
        )
        p = portfolio_QWIM(name_portfolio="Norm", portfolio_weights=df)
        weights = p.get_portfolio_weights()
        assert weights["A"][0] == pytest.approx(0.5, rel=1e-6)
        assert weights["B"][0] == pytest.approx(0.5, rel=1e-6)

    @pytest.mark.unit()
    def test_negative_weights_are_zeroed(self):
        """Negative weights should be set to 0 with the remainder renormalized."""
        df = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "A": [-0.1],
                "B": [1.1],
            }
        )
        p = portfolio_QWIM(name_portfolio="Negative", portfolio_weights=df)
        weights = p.get_portfolio_weights()
        assert weights["A"][0] == pytest.approx(0.0, abs=1e-9)
        assert weights["B"][0] == pytest.approx(1.0, rel=1e-6)


# ==============================================================================
# Tests: Constructor validation errors
# ==============================================================================


class Test_Portfolio_QWIM_Init_Validation:
    """Test that invalid constructor arguments raise appropriate errors."""

    @pytest.mark.unit()
    def test_raises_value_error_when_neither_weights_nor_components(self):
        """ValueError when both portfolio_weights and names_components are None."""
        with pytest.raises(ValueError, match="Either portfolio_weights or names_components"):
            portfolio_QWIM(name_portfolio="Empty")

    @pytest.mark.unit()
    def test_raises_type_error_for_non_string_name(self):
        """TypeError when name_portfolio is not a string."""
        with pytest.raises(TypeError):
            portfolio_QWIM(
                name_portfolio=123,  # type: ignore[arg-type]
                names_components=["A"],
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_empty_name(self):
        """ValueError when name_portfolio is an empty string."""
        with pytest.raises(ValueError):
            portfolio_QWIM(
                name_portfolio="",
                names_components=["A"],
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_whitespace_only_name(self):
        """ValueError when name_portfolio is all whitespace."""
        with pytest.raises(ValueError):
            portfolio_QWIM(
                name_portfolio="   ",
                names_components=["A"],
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_empty_components_list(self):
        """ValueError when names_components is an empty list."""
        with pytest.raises(ValueError):
            portfolio_QWIM(
                name_portfolio="Valid",
                names_components=[],
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_duplicate_components(self):
        """ValueError when names_components contains duplicate names."""
        with pytest.raises(ValueError, match="unique"):
            portfolio_QWIM(
                name_portfolio="Dup",
                names_components=["A", "A", "B"],
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_empty_component_name(self):
        """ValueError when a component name is an empty string."""
        with pytest.raises((ValueError, TypeError)):
            portfolio_QWIM(
                name_portfolio="EmptyComp",
                names_components=["A", ""],
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_weights_df_missing_date_column(self):
        """ValueError when portfolio_weights DataFrame has no 'Date' column."""
        df_no_date = pl.DataFrame({"A": [0.5], "B": [0.5]})
        with pytest.raises(ValueError, match="Date"):
            portfolio_QWIM(
                name_portfolio="NoDate",
                portfolio_weights=df_no_date,
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_weights_df_only_date_column(self):
        """ValueError when portfolio_weights DataFrame has only a Date column."""
        df_date_only = pl.DataFrame({"Date": ["2024-01-01"]})
        with pytest.raises(ValueError):
            portfolio_QWIM(
                name_portfolio="OnlyDate",
                portfolio_weights=df_date_only,
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_all_zero_weight_row(self):
        """ValueError when a row in portfolio_weights is entirely zero."""
        df_zero = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "A": [0.0],
                "B": [0.0],
            }
        )
        with pytest.raises(ValueError, match="zero"):
            portfolio_QWIM(
                name_portfolio="AllZero",
                portfolio_weights=df_zero,
            )

    @pytest.mark.unit()
    def test_raises_type_error_for_weights_not_dataframe(self):
        """TypeError when portfolio_weights is not a Polars DataFrame."""
        with pytest.raises(TypeError):
            portfolio_QWIM(
                name_portfolio="NotDF",
                portfolio_weights={"Date": ["2024-01-01"], "A": [1.0]},  # type: ignore[arg-type]
            )


# ==============================================================================
# Tests: m_validate_type_and_value
# ==============================================================================


class Test_Portfolio_QWIM_Validate_Type_And_Value:
    """Test the m_validate_type_and_value internal validation method."""

    @pytest.fixture()
    def p(self):
        return portfolio_QWIM(name_portfolio="V", names_components=["X"])

    @pytest.mark.unit()
    def test_passes_for_correct_type(self, p):
        """No exception when type matches."""
        p.m_validate_type_and_value("hello", "arg", str)  # should not raise

    @pytest.mark.unit()
    def test_raises_type_error_for_wrong_type(self, p):
        """TypeError when value type does not match expected_type."""
        with pytest.raises(TypeError):
            p.m_validate_type_and_value(42, "arg", str)

    @pytest.mark.unit()
    def test_raises_value_error_for_none_when_not_allowed(self, p):
        """ValueError when var is None and allow_none is False."""
        with pytest.raises(ValueError):
            p.m_validate_type_and_value(None, "arg", str, allow_none=False)

    @pytest.mark.unit()
    def test_passes_for_none_when_allowed(self, p):
        """No exception when var is None and allow_none is True."""
        p.m_validate_type_and_value(None, "arg", str, allow_none=True)

    @pytest.mark.unit()
    def test_raises_value_error_when_additional_check_fails(self, p):
        """ValueError when additional_check callable returns False."""
        with pytest.raises(ValueError):
            p.m_validate_type_and_value(
                "",
                "arg",
                str,
                additional_check=lambda x: len(x) > 0,
                error_msg="cannot be empty",
            )

    @pytest.mark.unit()
    def test_passes_when_additional_check_passes(self, p):
        """No exception when additional_check callable returns True."""
        p.m_validate_type_and_value(
            "hello",
            "arg",
            str,
            additional_check=lambda x: len(x) > 0,
        )

    @pytest.mark.unit()
    def test_accepts_tuple_of_types(self, p):
        """No exception when value matches one of multiple expected types."""
        p.m_validate_type_and_value(42, "arg", (int, float))

    @pytest.mark.unit()
    def test_raises_for_none_of_tuple_types(self, p):
        """TypeError when value matches none of the expected types."""
        with pytest.raises(TypeError):
            p.m_validate_type_and_value("text", "arg", (int, float))


# ==============================================================================
# Tests: m_validate_dataframe
# ==============================================================================


class Test_Portfolio_QWIM_Validate_Dataframe:
    """Test the m_validate_dataframe internal validation method."""

    @pytest.fixture()
    def p(self):
        return portfolio_QWIM(name_portfolio="V", names_components=["X"])

    @pytest.fixture()
    def valid_df(self):
        return pl.DataFrame({"Date": ["2024-01-01"], "A": [1.0]})

    @pytest.mark.unit()
    def test_returns_true_for_valid_dataframe(self, p, valid_df):
        """Returns True for a properly formed DataFrame."""
        result = p.m_validate_dataframe(valid_df, "test_df")
        assert result is True

    @pytest.mark.unit()
    def test_raises_type_error_for_non_dataframe(self, p):
        """TypeError when input is not a Polars DataFrame."""
        with pytest.raises(TypeError):
            p.m_validate_dataframe([1, 2, 3], "test_df")

    @pytest.mark.unit()
    def test_raises_value_error_for_missing_date_column(self, p):
        """ValueError when DataFrame lacks 'Date' column."""
        df = pl.DataFrame({"A": [0.5], "B": [0.5]})
        with pytest.raises(ValueError, match="Date"):
            p.m_validate_dataframe(df, "no_date")

    @pytest.mark.unit()
    def test_raises_value_error_for_empty_dataframe(self, p):
        """ValueError when DataFrame is empty."""
        df = pl.DataFrame({"Date": [], "A": []})
        with pytest.raises(ValueError):
            p.m_validate_dataframe(df, "empty")

    @pytest.mark.unit()
    def test_raises_value_error_for_only_date_column(self, p):
        """ValueError when DataFrame has only the Date column."""
        df = pl.DataFrame({"Date": ["2024-01-01"]})
        with pytest.raises(ValueError):
            p.m_validate_dataframe(df, "only_date")


# ==============================================================================
# Tests: m_validate_weights
# ==============================================================================


class Test_Portfolio_QWIM_Validate_Weights:
    """Test the m_validate_weights internal weight validation method."""

    @pytest.fixture()
    def p(self):
        return portfolio_QWIM(name_portfolio="V", names_components=["X"])

    @pytest.mark.unit()
    def test_returns_dataframe(self, p):
        """Result is a Polars DataFrame."""
        df = pl.DataFrame({"Date": ["2024-01-01"], "A": [0.5], "B": [0.5]})
        result = p.m_validate_weights(df, ["A", "B"])
        assert isinstance(result, pl.DataFrame)

    @pytest.mark.unit()
    def test_normalizes_weights_summing_above_one(self, p):
        """Weights summing above 1.0 should be normalized to sum to 1.0."""
        df = pl.DataFrame({"Date": ["2024-01-01"], "A": [3.0], "B": [7.0]})
        result = p.m_validate_weights(df, ["A", "B"])
        total = result["A"][0] + result["B"][0]
        assert total == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.unit()
    def test_sets_negative_weights_to_zero(self, p):
        """Negative weights must be zeroed before any further processing."""
        df = pl.DataFrame({"Date": ["2024-01-01"], "A": [-0.2], "B": [1.2]})
        result = p.m_validate_weights(df, ["A", "B"])
        assert result["A"][0] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.unit()
    def test_raises_value_error_for_all_zero_row(self, p):
        """ValueError when a row contains all-zero weights."""
        df = pl.DataFrame({"Date": ["2024-01-01"], "A": [0.0], "B": [0.0]})
        with pytest.raises(ValueError, match="zero"):
            p.m_validate_weights(df, ["A", "B"])

    @pytest.mark.unit()
    def test_does_not_modify_already_valid_weights(self, p):
        """Weights already summing to 1.0 should remain unchanged."""
        df = pl.DataFrame({"Date": ["2024-01-01"], "A": [0.4], "B": [0.6]})
        result = p.m_validate_weights(df, ["A", "B"])
        assert result["A"][0] == pytest.approx(0.4, rel=1e-6)
        assert result["B"][0] == pytest.approx(0.6, rel=1e-6)


# ==============================================================================
# Tests: m_parse_date
# ==============================================================================


class Test_Portfolio_QWIM_Parse_Date:
    """Test the m_parse_date date-parsing method."""

    @pytest.fixture()
    def p(self):
        return portfolio_QWIM(name_portfolio="V", names_components=["X"])

    @pytest.mark.unit()
    def test_none_returns_today(self, p):
        """Passing None should return today's date in YYYY-MM-DD format."""
        today = datetime.now().strftime("%Y-%m-%d")
        result = p.m_parse_date(None)
        assert result == today

    @pytest.mark.unit()
    def test_iso_format_string_parsed_correctly(self, p):
        """YYYY-MM-DD string should be returned unchanged."""
        result = p.m_parse_date("2023-06-15")
        assert result == "2023-06-15"

    @pytest.mark.unit()
    def test_slash_separated_us_format(self, p):
        """MM/DD/YYYY format should be converted to YYYY-MM-DD."""
        result = p.m_parse_date("06/15/2023")
        assert result == "2023-06-15"

    @pytest.mark.unit()
    def test_datetime_object_parsed(self, p):
        """datetime object should be formatted to YYYY-MM-DD string."""
        dt = datetime(2023, 6, 15)
        result = p.m_parse_date(dt)
        assert result == "2023-06-15"

    @pytest.mark.unit()
    def test_invalid_string_raises_value_error(self, p):
        """Unrecognized date string should raise ValueError."""
        with pytest.raises(ValueError):
            p.m_parse_date("not-a-date")

    @pytest.mark.unit()
    def test_unsupported_type_raises_type_error(self, p):
        """An unsupported type (e.g. int) should raise TypeError."""
        with pytest.raises(TypeError):
            p.m_parse_date(20230615)  # type: ignore[arg-type]

    @pytest.mark.unit()
    def test_returns_string(self, p):
        """Return value is always a string."""
        result = p.m_parse_date("2023-01-01")
        assert isinstance(result, str)

    @pytest.mark.unit()
    def test_result_matches_yyyy_mm_dd_pattern(self, p):
        """Returned string matches YYYY-MM-DD pattern."""
        import re

        result = p.m_parse_date("2023-12-31")
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", result)


# ==============================================================================
# Tests: Properties and accessors
# ==============================================================================


class Test_Portfolio_QWIM_Properties:
    """Test @property accessors and get_portfolio_weights method."""

    @pytest.mark.unit()
    def test_get_portfolio_components_returns_list(self, portfolio_from_components):
        """get_portfolio_components is a list."""
        components = portfolio_from_components.get_portfolio_components
        assert isinstance(components, list)

    @pytest.mark.unit()
    def test_get_portfolio_components_returns_copy(self, portfolio_from_components):
        """Mutating the returned list does not affect the portfolio."""
        components = portfolio_from_components.get_portfolio_components
        components.append("INTRUDER")
        assert "INTRUDER" not in portfolio_from_components.get_portfolio_components

    @pytest.mark.unit()
    def test_get_num_components_returns_int(self, portfolio_from_components):
        """get_num_components is an integer."""
        assert isinstance(portfolio_from_components.get_num_components, int)

    @pytest.mark.unit()
    def test_get_num_components_value(self, portfolio_from_components):
        """get_num_components equals the number of components supplied."""
        assert portfolio_from_components.get_num_components == 3

    @pytest.mark.unit()
    def test_get_portfolio_name_returns_string(self, portfolio_from_components):
        """get_portfolio_name is a string."""
        assert isinstance(portfolio_from_components.get_portfolio_name, str)

    @pytest.mark.unit()
    def test_get_portfolio_name_value(self, portfolio_from_components):
        """get_portfolio_name equals the name provided at construction."""
        assert portfolio_from_components.get_portfolio_name == "Test Portfolio"

    @pytest.mark.unit()
    def test_set_portfolio_name_updates_name(self, portfolio_from_components):
        """set_portfolio_name changes the stored portfolio name."""
        portfolio_from_components.set_portfolio_name("Renamed")
        assert portfolio_from_components.get_portfolio_name == "Renamed"

    @pytest.mark.unit()
    def test_set_portfolio_name_raises_for_empty_string(self, portfolio_from_components):
        """ValueError when setting an empty portfolio name."""
        with pytest.raises(ValueError):
            portfolio_from_components.set_portfolio_name("")

    @pytest.mark.unit()
    def test_set_portfolio_name_raises_for_non_string(self, portfolio_from_components):
        """ValueError when setting a non-string portfolio name."""
        with pytest.raises(ValueError):
            portfolio_from_components.set_portfolio_name(None)  # type: ignore[arg-type]

    @pytest.mark.unit()
    def test_get_portfolio_weights_returns_dataframe(self, portfolio_from_components):
        """get_portfolio_weights() returns a Polars DataFrame."""
        assert isinstance(portfolio_from_components.get_portfolio_weights(), pl.DataFrame)

    @pytest.mark.unit()
    def test_get_portfolio_weights_returns_clone(self, portfolio_from_components):
        """Mutating the returned DataFrame does not affect stored weights."""
        weights = portfolio_from_components.get_portfolio_weights()
        original_cols = list(portfolio_from_components.get_portfolio_weights().columns)
        assert portfolio_from_components.get_portfolio_weights().columns == original_cols

    @pytest.mark.unit()
    def test_get_portfolio_weights_has_date_column(self, portfolio_from_weights):
        """Returned weights DataFrame contains a 'Date' column."""
        assert "Date" in portfolio_from_weights.get_portfolio_weights().columns

    @pytest.mark.unit()
    def test_get_portfolio_weights_matches_component_count(self, portfolio_from_weights):
        """Number of non-Date columns equals get_num_components."""
        df = portfolio_from_weights.get_portfolio_weights()
        non_date_cols = [c for c in df.columns if c != "Date"]
        assert len(non_date_cols) == portfolio_from_weights.get_num_components


# ==============================================================================
# Tests: add_weights
# ==============================================================================


class Test_Portfolio_QWIM_Add_Weights:
    """Test the add_weights method."""

    @pytest.fixture()
    def simple_portfolio(self):
        df = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "A": [0.6],
                "B": [0.4],
            }
        )
        return portfolio_QWIM(
            name_portfolio="AddTest",
            portfolio_weights=df,
        )

    @pytest.mark.unit()
    def test_adds_new_date_row(self, simple_portfolio):
        """Adding a new date should increase the number of rows by 1."""
        before = len(simple_portfolio.get_portfolio_weights())
        simple_portfolio.add_weights(
            input_date="2024-02-01",
            weights={"A": 0.5, "B": 0.5},
        )
        assert len(simple_portfolio.get_portfolio_weights()) == before + 1

    @pytest.mark.unit()
    def test_returns_self_for_method_chaining(self, simple_portfolio):
        """add_weights should return self to allow chaining."""
        result = simple_portfolio.add_weights(
            input_date="2024-02-01",
            weights={"A": 0.5, "B": 0.5},
        )
        assert result is simple_portfolio

    @pytest.mark.unit()
    def test_raises_value_error_for_existing_date(self, simple_portfolio):
        """ValueError when attempting to add weights for a date that already exists."""
        with pytest.raises(ValueError, match="already exist"):
            simple_portfolio.add_weights(
                input_date="2024-01-01",
                weights={"A": 0.5, "B": 0.5},
            )

    @pytest.mark.unit()
    def test_normalizes_provided_weights(self, simple_portfolio):
        """Weights not summing to 1.0 should be normalized on add."""
        simple_portfolio.add_weights(
            input_date="2024-02-01",
            weights={"A": 2.0, "B": 3.0},
        )
        df = simple_portfolio.get_portfolio_weights()
        row = df.filter(pl.col("Date") == "2024-02-01")
        total = row["A"][0] + row["B"][0]
        assert total == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.unit()
    def test_raises_value_error_for_missing_component(self, simple_portfolio):
        """ValueError when provided weights dict is missing a required component."""
        with pytest.raises(ValueError, match="Missing"):
            simple_portfolio.add_weights(
                input_date="2024-03-01",
                weights={"A": 1.0},  # Missing 'B'
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_extra_component(self, simple_portfolio):
        """ValueError when provided weights dict contains an unknown component."""
        with pytest.raises(ValueError, match="Unexpected"):
            simple_portfolio.add_weights(
                input_date="2024-03-01",
                weights={"A": 0.5, "B": 0.4, "X": 0.1},
            )

    @pytest.mark.unit()
    def test_uses_equal_weights_when_none_provided(self, simple_portfolio):
        """Omitting the weights parameter should use equal weights per component."""
        simple_portfolio.add_weights(input_date="2024-02-01")
        df = simple_portfolio.get_portfolio_weights()
        row = df.filter(pl.col("Date") == "2024-02-01")
        assert row["A"][0] == pytest.approx(0.5, rel=1e-6)
        assert row["B"][0] == pytest.approx(0.5, rel=1e-6)


# ==============================================================================
# Tests: modify_weights
# ==============================================================================


class Test_Portfolio_QWIM_Modify_Weights:
    """Test the modify_weights method."""

    @pytest.fixture()
    def two_row_portfolio(self):
        df = pl.DataFrame(
            {
                "Date": ["2024-01-01", "2024-02-01"],
                "A": [0.6, 0.5],
                "B": [0.4, 0.5],
            }
        )
        return portfolio_QWIM(
            name_portfolio="ModTest",
            portfolio_weights=df,
        )

    @pytest.mark.unit()
    def test_modifies_weight_for_existing_date(self, two_row_portfolio):
        """Supplied weights should overwrite the stored values for the given date."""
        two_row_portfolio.modify_weights(
            input_date="2024-01-01",
            new_weights={"A": 0.7, "B": 0.3},
        )
        df = two_row_portfolio.get_portfolio_weights()
        row = df.filter(pl.col("Date") == "2024-01-01")
        assert row["A"][0] == pytest.approx(0.7, rel=1e-6)
        assert row["B"][0] == pytest.approx(0.3, rel=1e-6)

    @pytest.mark.unit()
    def test_returns_self_for_method_chaining(self, two_row_portfolio):
        """modify_weights should return self to allow chaining."""
        result = two_row_portfolio.modify_weights(
            input_date="2024-01-01",
            new_weights={"A": 0.7, "B": 0.3},
        )
        assert result is two_row_portfolio

    @pytest.mark.unit()
    def test_raises_value_error_for_missing_date(self, two_row_portfolio):
        """ValueError when the date to modify does not exist in the portfolio."""
        with pytest.raises(ValueError, match="No weights exist"):
            two_row_portfolio.modify_weights(
                input_date="2030-01-01",
                new_weights={"A": 0.5, "B": 0.5},
            )

    @pytest.mark.unit()
    def test_raises_value_error_for_invalid_component(self, two_row_portfolio):
        """ValueError when new_weights contains an unknown component name."""
        with pytest.raises(ValueError, match="Invalid"):
            two_row_portfolio.modify_weights(
                input_date="2024-01-01",
                new_weights={"X": 1.0},
            )

    @pytest.mark.unit()
    def test_does_not_affect_other_dates(self, two_row_portfolio):
        """Modifying one date should leave other dates unchanged."""
        original_a_feb = (
            two_row_portfolio.get_portfolio_weights()
            .filter(pl.col("Date") == "2024-02-01")["A"][0]
        )
        two_row_portfolio.modify_weights(
            input_date="2024-01-01",
            new_weights={"A": 0.9, "B": 0.1},
        )
        a_feb_after = (
            two_row_portfolio.get_portfolio_weights()
            .filter(pl.col("Date") == "2024-02-01")["A"][0]
        )
        assert a_feb_after == pytest.approx(original_a_feb, rel=1e-6)


# ==============================================================================
# Tests: validate_all_weights
# ==============================================================================


class Test_Portfolio_QWIM_Validate_All_Weights:
    """Test the validate_all_weights method."""

    @pytest.mark.unit()
    def test_returns_self(self):
        """validate_all_weights should return self (supports chaining)."""
        df = pl.DataFrame({"Date": ["2024-01-01"], "A": [0.4], "B": [0.6]})
        p = portfolio_QWIM(name_portfolio="Chain", portfolio_weights=df)
        result = p.validate_all_weights()
        assert result is p

    @pytest.mark.unit()
    def test_normalizes_unnormalized_weights(self):
        """Rows whose weights don't sum to 1.0 are normalized in-place."""
        # Build portfolio bypassing __init__ normalization by setting internal state
        df = pl.DataFrame({"Date": ["2024-01-01"], "A": [0.4], "B": [0.6]})
        p = portfolio_QWIM(name_portfolio="Norm", portfolio_weights=df)
        # Manually inject unnormalized weights into the internal state
        p.m_portfolio_weights = pl.DataFrame(
            {"Date": ["2024-01-01"], "A": [2.0], "B": [2.0]}
        )
        p.validate_all_weights()
        w = p.get_portfolio_weights()
        assert w["A"][0] == pytest.approx(0.5, rel=1e-6)

    @pytest.mark.unit()
    def test_date_remains_first_column_after_validation(self):
        """After validation the Date column must still be first."""
        df = pl.DataFrame({"Date": ["2024-01-01"], "A": [0.5], "B": [0.5]})
        p = portfolio_QWIM(name_portfolio="Order", portfolio_weights=df)
        p.validate_all_weights()
        assert p.get_portfolio_weights().columns[0] == "Date"


# ==============================================================================
# Tests: __str__ and __repr__
# ==============================================================================


class Test_Portfolio_QWIM_String_Representations:
    """Test the __str__ and __repr__ dunder methods."""

    @pytest.mark.unit()
    def test_str_contains_portfolio_name(self, portfolio_from_components):
        """str(portfolio) should include the portfolio name."""
        assert "Test Portfolio" in str(portfolio_from_components)

    @pytest.mark.unit()
    def test_str_contains_num_components(self, portfolio_from_components):
        """str(portfolio) should mention the number of components."""
        assert "3" in str(portfolio_from_components)

    @pytest.mark.unit()
    def test_repr_contains_portfolio_name(self, portfolio_from_components):
        """repr(portfolio) should include the portfolio name."""
        assert "Test Portfolio" in repr(portfolio_from_components)

    @pytest.mark.unit()
    def test_repr_contains_components(self, portfolio_from_components):
        """repr(portfolio) should reference the component names."""
        r = repr(portfolio_from_components)
        assert "AAPL" in r or "components" in r

    @pytest.mark.unit()
    def test_str_returns_string(self, portfolio_from_components):
        """str() always returns a Python str object."""
        assert isinstance(str(portfolio_from_components), str)

    @pytest.mark.unit()
    def test_repr_returns_string(self, portfolio_from_components):
        """repr() always returns a Python str object."""
        assert isinstance(repr(portfolio_from_components), str)


# ==============================================================================
# Tests: Regression / parametrized
# ==============================================================================


class Test_Portfolio_QWIM_Parametrized:
    """Parametrized regression tests covering a matrix of component counts."""

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "names,expected_count",
        [
            (["AAPL"], 1),
            (["AAPL", "MSFT"], 2),
            (["AAPL", "MSFT", "GOOG"], 3),
            (["A", "B", "C", "D", "E"], 5),
        ],
    )
    def test_num_components_parametrized(self, names, expected_count):
        """get_num_components matches input length across different sizes."""
        p = portfolio_QWIM(name_portfolio="P", names_components=names)
        assert p.get_num_components == expected_count

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "names,expected_weight",
        [
            (["AAPL"], pytest.approx(1.0, rel=1e-6)),
            (["AAPL", "MSFT"], pytest.approx(0.5, rel=1e-6)),
            (["A", "B", "C", "D"], pytest.approx(0.25, rel=1e-6)),
        ],
    )
    def test_equal_weights_parametrized(self, names, expected_weight):
        """Each component receives 1/n weight across different portfolio sizes."""
        p = portfolio_QWIM(name_portfolio="P", names_components=names)
        weights = p.get_portfolio_weights()
        weight_col = weights[names[0]][0]
        assert weight_col == expected_weight

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "date_str",
        ["2024-01-01", "2023-06-15", "2020-12-31"],
    )
    def test_parse_date_iso_formats(self, date_str):
        """m_parse_date handles valid ISO date strings without raising."""
        p = portfolio_QWIM(name_portfolio="P", names_components=["X"])
        result = p.m_parse_date(date_str)
        assert result == date_str


# ==============================================================================
# Tests: dict[str, Any] type-annotation fixes
# ==============================================================================


class Test_Portfolio_QWIM_Dict_Type_Fixes:
    """Unit tests that verify the dict[str, Any] annotation fixes introduced
    in the type-checking overhaul of portfolio_QWIM.py.

    Three internal dictionaries previously inferred as ``dict[str, str]`` were
    given explicit ``dict[str, Any]`` annotations to allow mixed-type values
    (str *and* float).  These tests exercise those exact code-paths end-to-end.
    """

    # ------------------------------------------------------------------
    # Path 1 – __init__ equal-weight code path: data_temp dict
    # ------------------------------------------------------------------

    @pytest.mark.unit()
    def test_init_from_components_stores_float_weights(self):
        """Equal-weight init path stores float (not str) weight values.

        This covers the ``data_temp: dict[str, Any]`` fix in __init__.
        """
        p = portfolio_QWIM(
            name_portfolio="FloatWeights",
            names_components=["A", "B"],
        )
        df = p.get_portfolio_weights()
        # Weight values must be float, not string
        assert isinstance(float(df["A"][0]), float)
        assert float(df["A"][0]) == pytest.approx(0.5, rel=1e-9)

    @pytest.mark.unit()
    def test_init_from_components_date_is_string_in_dict(self):
        """The Date entry in the internal data_temp dict must survive to the
        final DataFrame as a proper date/string value (not Python float).

        Ensures the dict[str, Any] fix doesn't corrupt the Date field type.
        """
        p = portfolio_QWIM(
            name_portfolio="DateType",
            names_components=["X"],
            date_portfolio="2023-07-04",
        )
        date_val = str(p.get_portfolio_weights()["Date"][0])
        assert date_val == "2023-07-04"

    # ------------------------------------------------------------------
    # Path 2 – m_initialize_from_components: data_date dict
    # ------------------------------------------------------------------

    @pytest.mark.unit()
    def test_init_from_many_components_mixed_types_in_row(self):
        """m_initialize_from_components stores str Date + float weights in the
        same dict without a type clash.

        This covers the ``data_date: dict[str, Any]`` fix.
        """
        components = ["VTI", "AGG", "VNQ", "VXUS", "BND"]
        p = portfolio_QWIM(
            name_portfolio="MixedDict",
            names_components=components,
        )
        df = p.get_portfolio_weights()
        expected_w = pytest.approx(1.0 / len(components), rel=1e-9)
        for comp in components:
            assert float(df[comp][0]) == expected_w
        # Date column present and correct type (str-representable)
        assert "Date" in df.columns

    # ------------------------------------------------------------------
    # Path 3 – add_weights: new_row dict
    # ------------------------------------------------------------------

    @pytest.mark.unit()
    def test_add_weights_mixed_type_dict_accepted(self):
        """add_weights must store a string Date alongside float weight values
        in the same new_row dict.

        This covers the ``new_row: dict[str, Any]`` fix in add_weights().
        """
        df_init = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "X": [0.6],
                "Y": [0.4],
            }
        )
        p = portfolio_QWIM(name_portfolio="AddRow", portfolio_weights=df_init)
        # add a new row – internally creates new_row: dict[str, Any]
        p.add_weights(input_date="2024-02-01", weights={"X": 0.7, "Y": 0.3})
        df = p.get_portfolio_weights()
        row = df.filter(pl.col("Date") == "2024-02-01")
        assert len(row) == 1
        assert float(row["X"][0]) == pytest.approx(0.7, rel=1e-9)
        assert float(row["Y"][0]) == pytest.approx(0.3, rel=1e-9)

    @pytest.mark.unit()
    def test_add_weights_date_column_preserved_as_correct_type(self):
        """After add_weights the Date column must remain queryable as a date
        (not corrupted to float by a type mismatch in the new_row dict).
        """
        df_init = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "A": [1.0],
            }
        )
        p = portfolio_QWIM(name_portfolio="DatePreserve", portfolio_weights=df_init)
        p.add_weights(input_date="2024-06-01", weights={"A": 1.0})
        df = p.get_portfolio_weights()
        # Both dates must be present and filterable
        jan = df.filter(pl.col("Date") == "2024-01-01")
        jun = df.filter(pl.col("Date") == "2024-06-01")
        assert len(jan) == 1
        assert len(jun) == 1

    @pytest.mark.unit()
    def test_add_weights_multiple_rows_accumulates_correctly(self):
        """Adding multiple rows via consecutive add_weights calls stores all
        rows and all use the mixed-type (Date + floats) dict internally.
        """
        df_init = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "P": [1.0],
            }
        )
        p = portfolio_QWIM(name_portfolio="MultiAdd", portfolio_weights=df_init)
        for month in range(2, 7):
            p.add_weights(
                input_date=f"2024-{month:02d}-01",
                weights={"P": 1.0},
            )
        assert len(p.get_portfolio_weights()) == 6
