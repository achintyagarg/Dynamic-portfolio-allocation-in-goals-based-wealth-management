"""Regression tests for the portfolios package.

These tests re-compute outputs from ``portfolio_QWIM`` and
``utils_portfolio`` and compare them against stored baseline artefacts
(parquet + JSON) to guard against silent numerical regressions introduced
by source-code changes.

Baseline artefacts are generated once by running::

    python tests/regression_data/portfolios/generate_baselines.py

and are checked in alongside the tests.  If intentional behaviour changes,
regenerate the baselines and commit the new files.

Test classes
------------
- ``Test_Regression_Portfolio_Object``
    Verifies that portfolio_QWIM construction and weight properties match
    the baseline.
- ``Test_Regression_Portfolio_Values``
    Verifies that ``calculate_portfolio_values`` produces the same numerical
    result as the stored baseline (first/last/min/max checks).
- ``Test_Regression_Benchmark_Values``
    Verifies that ``create_benchmark_portfolio_values`` reproduces the
    stored first and last values exactly.
- ``Test_Regression_Utils_Pipeline``
    End-to-end: calls ``get_sample_portfolio()`` and checks structural
    invariants guaranteed in every run (non-empty, positive values, Date
    column present).
- ``Test_Regression_Create_Custom_Portfolio``
    Regression guard for the ``date_portfolio=`` bug that was fixed; ensures
    future changes to ``create_custom_portfolio`` do not re-introduce the
    broken ``date=`` keyword argument.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest


# ==============================================================================
# Portfolio object regression
# ==============================================================================


@pytest.mark.regression()
class Test_Regression_Portfolio_Object:
    """Regression tests for portfolio_QWIM object construction from baselines."""

    def test_portfolio_has_correct_components(
        self,
        baseline_metadata,
        baseline_weights,
    ):
        """portfolio_QWIM built from baseline weights has the expected components."""
        from src.portfolios.portfolio_QWIM import portfolio_QWIM

        port = portfolio_QWIM(
            name_portfolio="Regression Portfolio",
            portfolio_weights=baseline_weights,
        )
        expected = sorted(baseline_metadata["components"])
        actual = sorted(port.get_portfolio_components)
        assert actual == expected

    def test_portfolio_num_components(
        self,
        baseline_metadata,
        baseline_weights,
    ):
        """get_num_components matches the number of components in the metadata."""
        from src.portfolios.portfolio_QWIM import portfolio_QWIM

        port = portfolio_QWIM(
            name_portfolio="Regression Portfolio",
            portfolio_weights=baseline_weights,
        )
        assert port.get_num_components == len(baseline_metadata["components"])

    def test_portfolio_weights_sum_to_one(
        self,
        baseline_weights,
    ):
        """All weight rows in the baseline must sum to 1.0 (±1 e-9)."""
        from src.portfolios.portfolio_QWIM import portfolio_QWIM

        port = portfolio_QWIM(
            name_portfolio="Regression Portfolio",
            portfolio_weights=baseline_weights,
        )
        df = port.get_portfolio_weights()
        weight_cols = [c for c in df.columns if c != "Date"]
        for i in range(len(df)):
            row_sum = sum(df[col][i] for col in weight_cols)
            assert row_sum == pytest.approx(1.0, abs=1e-9), (
                f"Row {i}: weights sum to {row_sum}, expected 1.0"
            )

    def test_baseline_equal_weights_reproduced(
        self,
        baseline_metadata,
        baseline_weights,
    ):
        """Each component in the baseline has the expected equal weight (0.2)."""
        from src.portfolios.portfolio_QWIM import portfolio_QWIM

        port = portfolio_QWIM(
            name_portfolio="Regression Portfolio",
            portfolio_weights=baseline_weights,
        )
        df = port.get_portfolio_weights()
        for comp, expected_weight in baseline_metadata["weights"].items():
            actual = float(df[comp][0])
            assert actual == pytest.approx(expected_weight, rel=1e-6), (
                f"Component {comp}: expected weight {expected_weight}, got {actual}"
            )

    def test_weights_dataframe_has_date_column(
        self,
        baseline_weights,
    ):
        """The weights DataFrame always starts with the Date column."""
        from src.portfolios.portfolio_QWIM import portfolio_QWIM

        port = portfolio_QWIM(
            name_portfolio="Regression Portfolio",
            portfolio_weights=baseline_weights,
        )
        assert port.get_portfolio_weights().columns[0] == "Date"


# ==============================================================================
# Portfolio values regression
# ==============================================================================


@pytest.mark.regression()
class Test_Regression_Portfolio_Values:
    """Regression tests verifying calculate_portfolio_values output."""

    @pytest.fixture(scope="class")
    def computed_values(
        self,
        baseline_weights,
        baseline_metadata,
    ) -> pl.DataFrame:
        """Recompute portfolio values from the baseline inputs (class-scoped)."""
        from datetime import timedelta  # noqa: PLC0415
        import random  # noqa: PLC0415

        from src.portfolios.portfolio_QWIM import portfolio_QWIM  # noqa: PLC0415
        from src.portfolios.utils_portfolio import calculate_portfolio_values  # noqa: PLC0415

        components = baseline_metadata["components"]
        n_rows = baseline_metadata["n_rows"]
        initial_value = baseline_metadata["initial_value"]

        # Reproduce the exact same price data used during baseline generation
        rng = random.Random(42)
        start = date(2023, 1, 2)
        date_list = [start + timedelta(days=i) for i in range(n_rows)]
        prices: dict[str, list] = {"Date": date_list}
        for comp in components:
            price = 100.0
            series: list[float] = []
            for _ in range(n_rows):
                price *= 1.0 + rng.uniform(-0.01, 0.01)
                series.append(round(price, 6))
            prices[comp] = series
        price_data = pl.DataFrame(prices).with_columns(pl.col("Date").cast(pl.Date))

        port = portfolio_QWIM(
            name_portfolio="Regression Portfolio",
            portfolio_weights=baseline_weights,
        )
        return calculate_portfolio_values(
            portfolio_obj=port,
            price_data=price_data,
            initial_value=initial_value,
        )

    def test_row_count_matches_baseline(
        self,
        computed_values,
        baseline_portfolio_values,
        baseline_metadata,
    ):
        """Recomputed DataFrame has the same number of rows as the baseline."""
        assert len(computed_values) == baseline_metadata["n_rows"]
        assert len(computed_values) == len(baseline_portfolio_values)

    def test_first_value_matches_baseline(
        self,
        computed_values,
        baseline_metadata,
    ):
        """First computed portfolio value equals the baseline first value."""
        expected = baseline_metadata["portfolio_values"]["first"]
        actual = float(computed_values["Portfolio_Value"][0])
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_last_value_matches_baseline(
        self,
        computed_values,
        baseline_metadata,
    ):
        """Last computed portfolio value matches the baseline last value."""
        expected = baseline_metadata["portfolio_values"]["last"]
        actual = float(computed_values["Portfolio_Value"][-1])
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_min_value_matches_baseline(
        self,
        computed_values,
        baseline_metadata,
    ):
        """Minimum computed portfolio value matches the baseline minimum."""
        expected = baseline_metadata["portfolio_values"]["min"]
        actual = float(computed_values["Portfolio_Value"].min())  # type: ignore[arg-type]
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_max_value_matches_baseline(
        self,
        computed_values,
        baseline_metadata,
    ):
        """Maximum computed portfolio value matches the baseline maximum."""
        expected = baseline_metadata["portfolio_values"]["max"]
        actual = float(computed_values["Portfolio_Value"].max())  # type: ignore[arg-type]
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_all_values_positive(self, computed_values):
        """Every portfolio value entry must be strictly positive."""
        min_val = computed_values["Portfolio_Value"].min()
        assert min_val is not None
        assert float(min_val) > 0.0

    def test_output_has_date_and_value_columns(self, computed_values):
        """Recomputed DataFrame must contain both Date and Portfolio_Value columns."""
        assert "Date" in computed_values.columns
        assert "Portfolio_Value" in computed_values.columns

    def test_initial_value_is_exactly_preserved(
        self,
        computed_values,
        baseline_metadata,
    ):
        """First Portfolio_Value must equal the initial_value supplied."""
        expected = baseline_metadata["initial_value"]
        actual = float(computed_values["Portfolio_Value"][0])
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_column_by_column_values_match(
        self,
        computed_values,
        baseline_portfolio_values,
    ):
        """Every individual Portfolio_Value matches the stored baseline (rel 1e-6)."""
        computed_list = computed_values["Portfolio_Value"].to_list()
        baseline_list = baseline_portfolio_values["Portfolio_Value"].to_list()
        assert len(computed_list) == len(baseline_list), (
            f"Row count mismatch: computed={len(computed_list)}, baseline={len(baseline_list)}"
        )
        for i, (c, b) in enumerate(zip(computed_list, baseline_list)):
            assert float(c) == pytest.approx(float(b), rel=1e-6), (  # type: ignore[arg-type]
                f"Row {i}: computed={c}, baseline={b}"
            )


# ==============================================================================
# Benchmark values regression
# ==============================================================================


@pytest.mark.regression()
class Test_Regression_Benchmark_Values:
    """Regression tests for create_benchmark_portfolio_values."""

    @pytest.fixture(scope="class")
    def computed_benchmark(
        self,
        baseline_portfolio_values,
    ) -> pl.DataFrame:
        """Recompute benchmark from the stored portfolio-values baseline."""
        from src.portfolios.utils_portfolio import create_benchmark_portfolio_values  # noqa: PLC0415

        return create_benchmark_portfolio_values(baseline_portfolio_values)

    def test_benchmark_first_value_matches(
        self,
        computed_benchmark,
        baseline_metadata,
    ):
        """First benchmark value equals the baseline first value."""
        expected = baseline_metadata["benchmark_values"]["first"]
        actual = float(computed_benchmark["Value"][0])
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_benchmark_last_value_matches(
        self,
        computed_benchmark,
        baseline_metadata,
    ):
        """Last benchmark value matches the stored baseline."""
        expected = baseline_metadata["benchmark_values"]["last"]
        actual = float(computed_benchmark["Value"][-1])
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_benchmark_row_count_matches(
        self,
        computed_benchmark,
        baseline_benchmark_values,
    ):
        """Benchmark row count matches the stored baseline."""
        assert len(computed_benchmark) == len(baseline_benchmark_values)

    def test_benchmark_reproducible(
        self,
        baseline_portfolio_values,
    ):
        """Two calls with identical input produce identical benchmark values."""
        from src.portfolios.utils_portfolio import create_benchmark_portfolio_values  # noqa: PLC0415

        b1 = create_benchmark_portfolio_values(baseline_portfolio_values)
        b2 = create_benchmark_portfolio_values(baseline_portfolio_values)
        assert b1["Value"].to_list() == b2["Value"].to_list()

    def test_benchmark_differs_from_portfolio(
        self,
        computed_benchmark,
        baseline_portfolio_values,
    ):
        """Benchmark series must not be identical to the portfolio values series."""
        port_list = baseline_portfolio_values["Portfolio_Value"].to_list()
        bench_list = computed_benchmark["Value"].to_list()
        assert port_list != bench_list

    def test_column_by_column_benchmark_values(
        self,
        computed_benchmark,
        baseline_benchmark_values,
    ):
        """Every individual benchmark Value matches the stored baseline (rel 1e-6)."""
        computed_list = computed_benchmark["Value"].to_list()
        baseline_list = baseline_benchmark_values["Value"].to_list()
        for i, (c, b) in enumerate(zip(computed_list, baseline_list)):
            assert float(c) == pytest.approx(float(b), rel=1e-6), (  # type: ignore[arg-type]
                f"Row {i}: computed={c}, baseline={b}"
            )


# ==============================================================================
# End-to-end pipeline regression
# ==============================================================================


@pytest.mark.regression()
class Test_Regression_Utils_Pipeline:
    """Structural regression tests for the get_sample_portfolio() pipeline."""

    def test_get_sample_portfolio_returns_tuple(self):
        """get_sample_portfolio returns a 3-tuple."""
        from src.portfolios.utils_portfolio import get_sample_portfolio  # noqa: PLC0415

        result = get_sample_portfolio()
        assert len(result) == 3

    def test_portfolio_object_is_correct_type(self):
        """First element is a portfolio_QWIM instance."""
        from src.portfolios.portfolio_QWIM import portfolio_QWIM  # noqa: PLC0415
        from src.portfolios.utils_portfolio import get_sample_portfolio  # noqa: PLC0415

        port, _, _ = get_sample_portfolio()
        assert isinstance(port, portfolio_QWIM)

    def test_etf_data_is_dataframe(self):
        """Second element (ETF prices) is a Polars DataFrame."""
        from src.portfolios.utils_portfolio import get_sample_portfolio  # noqa: PLC0415

        _, etf, _ = get_sample_portfolio()
        assert isinstance(etf, pl.DataFrame)

    def test_portfolio_values_is_dataframe(self):
        """Third element (portfolio values) is a Polars DataFrame."""
        from src.portfolios.utils_portfolio import get_sample_portfolio  # noqa: PLC0415

        _, _, pv = get_sample_portfolio()
        assert isinstance(pv, pl.DataFrame)

    def test_portfolio_values_not_empty(self):
        """Portfolio values DataFrame is non-empty."""
        from src.portfolios.utils_portfolio import get_sample_portfolio  # noqa: PLC0415

        _, _, pv = get_sample_portfolio()
        assert len(pv) > 0

    def test_portfolio_values_all_positive(self):
        """All portfolio values are strictly positive."""
        from src.portfolios.utils_portfolio import get_sample_portfolio  # noqa: PLC0415

        _, _, pv = get_sample_portfolio()
        min_val = pv["Portfolio_Value"].min()
        assert min_val is not None and float(min_val) > 0.0

    def test_portfolio_values_has_required_columns(self):
        """Portfolio values has Date and Portfolio_Value columns."""
        from src.portfolios.utils_portfolio import get_sample_portfolio  # noqa: PLC0415

        _, _, pv = get_sample_portfolio()
        assert "Date" in pv.columns
        assert "Portfolio_Value" in pv.columns

    def test_portfolio_has_at_least_one_component(self):
        """Portfolio has at least one component after full pipeline."""
        from src.portfolios.utils_portfolio import get_sample_portfolio  # noqa: PLC0415

        port, _, _ = get_sample_portfolio()
        assert port.get_num_components >= 1

    def test_weights_sum_to_one(self):
        """All weight rows in the returned portfolio sum to 1.0."""
        from src.portfolios.utils_portfolio import get_sample_portfolio  # noqa: PLC0415

        port, _, _ = get_sample_portfolio()
        df = port.get_portfolio_weights()
        weight_cols = [c for c in df.columns if c != "Date"]
        for i in range(len(df)):
            row_sum = sum(df[col][i] for col in weight_cols)
            assert row_sum == pytest.approx(1.0, abs=1e-9), (
                f"Row {i}: weights sum to {row_sum}"
            )


# ==============================================================================
# create_custom_portfolio regression (guards the date_portfolio= bug fix)
# ==============================================================================


@pytest.mark.regression()
class Test_Regression_Create_Custom_Portfolio:
    """Regression tests guarding the create_custom_portfolio() bug fix.

    Before the fix the function used ``date=date`` as the keyword argument
    to ``portfolio_QWIM.__init__``.  The correct parameter name is
    ``date_portfolio``.  These tests ensure this regression never recurs.
    """

    def test_creates_portfolio_without_date(self):
        """create_custom_portfolio works when no date is provided."""
        from src.portfolios.portfolio_QWIM import portfolio_QWIM  # noqa: PLC0415
        from src.portfolios.utils_portfolio import create_custom_portfolio  # noqa: PLC0415

        result = create_custom_portfolio(["VTI", "AGG", "BND"])
        assert isinstance(result, portfolio_QWIM)

    def test_creates_portfolio_with_date(self):
        """create_custom_portfolio correctly passes date_portfolio= to constructor.

        This specifically guards against the regression where ``date=`` was
        used (an invalid keyword argument) instead of ``date_portfolio=``.
        """
        from src.portfolios.portfolio_QWIM import portfolio_QWIM  # noqa: PLC0415
        from src.portfolios.utils_portfolio import create_custom_portfolio  # noqa: PLC0415

        result = create_custom_portfolio(["VTI", "AGG"], date="2024-03-15")
        assert isinstance(result, portfolio_QWIM)
        # The stored date must equal the supplied date (not default to today)
        weights_df = result.get_portfolio_weights()
        stored_date = str(weights_df["Date"][0])
        assert stored_date == "2024-03-15"

    def test_components_stored_in_portfolio(self):
        """Returned portfolio stores the exact components supplied."""
        from src.portfolios.utils_portfolio import create_custom_portfolio  # noqa: PLC0415

        components = ["SPY", "QQQ", "IWM"]
        result = create_custom_portfolio(components)
        assert sorted(result.get_portfolio_components) == sorted(components)

    def test_equal_weights_assigned(self):
        """Components receive equal weights when no weights are specified."""
        from src.portfolios.utils_portfolio import create_custom_portfolio  # noqa: PLC0415

        result = create_custom_portfolio(["A", "B", "C", "D"])
        df = result.get_portfolio_weights()
        for col in ["A", "B", "C", "D"]:
            assert float(df[col][0]) == pytest.approx(0.25, rel=1e-6)

    def test_raises_value_error_for_empty_components(self):
        """ValueError is raised when an empty component list is supplied."""
        from src.portfolios.utils_portfolio import create_custom_portfolio  # noqa: PLC0415

        with pytest.raises(ValueError):
            create_custom_portfolio([])

    @pytest.mark.parametrize(
        "components",
        [
            (["VTI"],),
            (["VTI", "AGG"],),
            (["VTI", "AGG", "VNQ"],),
            (["VTI", "AGG", "VNQ", "VXUS"],),
        ],
    )
    def test_num_components_matches_input_length(self, components):
        """get_num_components equals len(components) for various list sizes."""
        from src.portfolios.utils_portfolio import create_custom_portfolio  # noqa: PLC0415

        result = create_custom_portfolio(components[0])
        assert result.get_num_components == len(components[0])
