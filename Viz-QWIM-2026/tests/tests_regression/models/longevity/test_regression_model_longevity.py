"""Regression tests for QWIM longevity models (Constant and Gompertz).

Every test in this module compares the **current** model output against
a stored Parquet baseline.  If a test fails, either the code introduced
an unintended change or the baseline must be intentionally regenerated
with ``generate_baselines.py``.

Test organisation
-----------------
* :class:`Class_Test_Regression_Constant_Default`
    — constant model with constructor qx, empty fit.
* :class:`Class_Test_Regression_Constant_Fitted`
    — constant model fitted to historical data.
* :class:`Class_Test_Regression_Constant_Historical_Data`
    — canonical input data round-trip.
* :class:`Class_Test_Regression_Constant_Life_Expectancy`
    — life expectancy under constant mortality.
* :class:`Class_Test_Regression_Constant_Survival`
    — survival probabilities under constant mortality.
* :class:`Class_Test_Regression_Gompertz_Fitted_Parameters`
    — Gompertz OLS parameter recovery.
* :class:`Class_Test_Regression_Gompertz_Predictions`
    — Gompertz mortality-table predictions.
* :class:`Class_Test_Regression_Gompertz_Life_Expectancy`
    — Gompertz life expectancy at various ages.
* :class:`Class_Test_Regression_Gompertz_Survival`
    — Gompertz analytical survival probabilities.
* :class:`Class_Test_Regression_Gompertz_Force_Of_Mortality`
    — Gompertz hazard rate at selected ages.
* :class:`Class_Test_Regression_Cross_Model_Consistency`
    — sanity checks across both models.

Usage
-----
    pytest tests/tests_regression/models/longevity/ -v -m regression
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from src.models.longevity.model_longevity_base import Longevity_Model_Status


if TYPE_CHECKING:
    from src.models.longevity.model_longevity_constant import Longevity_Model_Constant
    from src.models.longevity.model_longevity_standard import Longevity_Model_Standard

from .conftest import (
    CONSTANT_DEFAULT_QX,
    CONSTANT_N_AGES,
    CONSTANT_START_AGE,
    FORCE_OF_MORTALITY_AGES,
    GOMPERTZ_N_AGES_LONG,
    GOMPERTZ_N_AGES_SHORT,
    GOMPERTZ_START_AGE,
    LIFE_EXPECTANCY_AGES,
    SURVIVAL_AGES,
    SURVIVAL_HORIZONS,
    load_baseline,
)


# ======================================================================
# Tolerances
# ======================================================================

# Deterministic (constant model): should be exact up to floating-point
_TOL_DETERMINISTIC: float = 1e-10

# Numerical (Gompertz OLS fit on exact data): very tight
_TOL_NUMERICAL: float = 1e-8

# Moderate (derived quantities — life expectancy sums, etc.)
_TOL_MODERATE: float = 1e-6


# ======================================================================
# Helpers
# ======================================================================


def _assert_dataframe_matches_baseline(
    actual: pl.DataFrame,
    baseline: pl.DataFrame,
    tolerance: float,
    *,
    skip_columns: list[str] | None = None,
) -> None:
    """Assert that every numeric column in *actual* matches *baseline*.

    Parameters
    ----------
    actual : pl.DataFrame
        Freshly computed DataFrame.
    baseline : pl.DataFrame
        Stored Parquet reference.
    tolerance : float
        Maximum allowed absolute difference for numeric columns.
    skip_columns : list[str] or None
        Column names to exclude from the comparison.
    """
    skip = set(skip_columns or [])

    assert actual.shape == baseline.shape, (
        f"Shape mismatch: actual {actual.shape} vs baseline {baseline.shape}"
    )

    for col_name in baseline.columns:
        if col_name in skip:
            continue

        actual_col = actual[col_name]
        baseline_col = baseline[col_name]

        if baseline_col.dtype in (pl.Float64, pl.Float32):
            actual_arr = actual_col.to_numpy()
            baseline_arr = baseline_col.to_numpy()
            np.testing.assert_allclose(
                actual_arr,
                baseline_arr,
                atol=tolerance,
                rtol=0.0,
                err_msg=f"Column '{col_name}' mismatch (atol={tolerance})",
            )
        elif baseline_col.dtype in (pl.Int64, pl.Int32, pl.Int8, pl.UInt8):
            assert actual_col.to_list() == baseline_col.to_list(), (
                f"Integer column '{col_name}' mismatch"
            )
        else:
            assert actual_col.to_list() == baseline_col.to_list(), f"Column '{col_name}' mismatch"


# ======================================================================
# Constant model — Default (constructor qx, empty fit)
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Constant_Default:
    """Regression tests for the constant model with constructor qx=0.02."""

    def Test_Predict_Matches_Baseline(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Prediction DataFrame must match the stored baseline."""
        baseline = load_baseline("constant_predict_default")
        actual = fitted_constant_default.predict(
            n_ages=CONSTANT_N_AGES,
            start_age=CONSTANT_START_AGE,
        )
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_DETERMINISTIC)

    def Test_Summary_Matches_Baseline(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Summary statistics must match the stored baseline."""
        baseline = load_baseline("constant_summary_default")
        pred = fitted_constant_default.predict(
            n_ages=CONSTANT_N_AGES,
            start_age=CONSTANT_START_AGE,
        )
        actual = fitted_constant_default.get_summary_statistics(pred)
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_DETERMINISTIC)

    def Test_Parameters_Match_Baseline(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Stored parameters must match the baseline."""
        baseline = load_baseline("constant_parameters_default")
        actual = pl.DataFrame(
            {key: [val] for key, val in fitted_constant_default.parameters.items()},
        )
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_DETERMINISTIC)

    def Test_Qx_Equals_Constructor_Value(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Fitted qx must equal the constructor default (0.02)."""
        assert fitted_constant_default.parameters["qx"] == pytest.approx(
            CONSTANT_DEFAULT_QX,
            abs=_TOL_DETERMINISTIC,
        )

    def Test_All_Qx_Are_Constant(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Every row's qx must be the same value."""
        pred = fitted_constant_default.predict(
            n_ages=CONSTANT_N_AGES,
            start_age=CONSTANT_START_AGE,
        )
        qx_vals = pred["qx"].to_list()
        assert all(
            val == pytest.approx(CONSTANT_DEFAULT_QX, abs=_TOL_DETERMINISTIC) for val in qx_vals
        )

    def Test_Prediction_Shape(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Prediction shape must be (CONSTANT_N_AGES, 2)."""
        pred = fitted_constant_default.predict(
            n_ages=CONSTANT_N_AGES,
            start_age=CONSTANT_START_AGE,
        )
        assert pred.shape == (CONSTANT_N_AGES, 2)

    def Test_Status_Is_Fitted(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Model status must be FITTED."""
        assert fitted_constant_default.is_fitted

    def Test_Ages_Start_At_Expected_Value(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Age column must start at CONSTANT_START_AGE."""
        pred = fitted_constant_default.predict(
            n_ages=CONSTANT_N_AGES,
            start_age=CONSTANT_START_AGE,
        )
        assert pred["Age"][0] == CONSTANT_START_AGE
        assert pred["Age"][-1] == CONSTANT_START_AGE + CONSTANT_N_AGES - 1


# ======================================================================
# Constant model — Fitted to historical data
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Constant_Fitted:
    """Regression tests for the constant model fitted to historical qx data."""

    def Test_Predict_Matches_Baseline(
        self,
        fitted_constant_historical: Longevity_Model_Constant,
    ) -> None:
        """Prediction DataFrame must match the stored baseline."""
        baseline = load_baseline("constant_predict_fitted")
        actual = fitted_constant_historical.predict(
            n_ages=CONSTANT_N_AGES,
            start_age=CONSTANT_START_AGE,
        )
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_DETERMINISTIC)

    def Test_Summary_Matches_Baseline(
        self,
        fitted_constant_historical: Longevity_Model_Constant,
    ) -> None:
        """Summary statistics must match the stored baseline."""
        baseline = load_baseline("constant_summary_fitted")
        pred = fitted_constant_historical.predict(
            n_ages=CONSTANT_N_AGES,
            start_age=CONSTANT_START_AGE,
        )
        actual = fitted_constant_historical.get_summary_statistics(pred)
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_DETERMINISTIC)

    def Test_Parameters_Match_Baseline(
        self,
        fitted_constant_historical: Longevity_Model_Constant,
    ) -> None:
        """Stored parameters must match the baseline."""
        baseline = load_baseline("constant_parameters_fitted")
        actual = pl.DataFrame(
            {key: [val] for key, val in fitted_constant_historical.parameters.items()},
        )
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_DETERMINISTIC)

    def Test_Fitted_Qx_Equals_Mean_Of_Historical(
        self,
        fitted_constant_historical: Longevity_Model_Constant,
        constant_historical_qx: pl.DataFrame,
    ) -> None:
        """Fitted qx must equal the arithmetic mean of the historical data."""
        expected_qx = float(np.mean(constant_historical_qx["qx"].to_numpy()))
        assert fitted_constant_historical.parameters["qx"] == pytest.approx(
            expected_qx,
            abs=_TOL_DETERMINISTIC,
        )

    def Test_Status_Is_Fitted(
        self,
        fitted_constant_historical: Longevity_Model_Constant,
    ) -> None:
        """Model status must be FITTED."""
        assert fitted_constant_historical.is_fitted

    def Test_All_Qx_Are_Constant(
        self,
        fitted_constant_historical: Longevity_Model_Constant,
    ) -> None:
        """Every qx must equal the fitted value."""
        pred = fitted_constant_historical.predict(
            n_ages=CONSTANT_N_AGES,
            start_age=CONSTANT_START_AGE,
        )
        fitted_qx = fitted_constant_historical.parameters["qx"]
        qx_vals = pred["qx"].to_list()
        assert all(val == pytest.approx(fitted_qx, abs=_TOL_DETERMINISTIC) for val in qx_vals)


# ======================================================================
# Constant model — Historical data round-trip
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Constant_Historical_Data:
    """Verify the canonical historical data matches the stored baseline."""

    def Test_Historical_Data_Matches_Baseline(
        self,
        constant_historical_qx: pl.DataFrame,
    ) -> None:
        """Input data must match the Parquet baseline exactly."""
        baseline = load_baseline("constant_historical_qx")
        _assert_dataframe_matches_baseline(
            constant_historical_qx,
            baseline,
            _TOL_DETERMINISTIC,
        )

    def Test_Historical_Data_Shape(
        self,
        constant_historical_qx: pl.DataFrame,
    ) -> None:
        """Historical data must have 5 rows and 2 columns."""
        assert constant_historical_qx.shape == (5, 2)


# ======================================================================
# Constant model — Life expectancy
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Constant_Life_Expectancy:
    """Regression tests for constant-model life expectancy."""

    def Test_Life_Expectancy_Matches_Baseline(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Life expectancy at canonical ages must match the stored baseline."""
        baseline = load_baseline("constant_life_expectancy_default")

        for idx_row in range(len(baseline)):
            age = int(baseline["Age"][idx_row])
            expected_le = float(baseline["Life_Expectancy"][idx_row])
            actual_le = fitted_constant_default.get_life_expectancy(age)
            assert actual_le == pytest.approx(expected_le, abs=_TOL_DETERMINISTIC), (
                f"Life expectancy mismatch at age {age}"
            )

    def Test_Life_Expectancy_Is_Age_Independent(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Under constant mortality, life expectancy does not depend on age."""
        le_0 = fitted_constant_default.get_life_expectancy(0)
        le_65 = fitted_constant_default.get_life_expectancy(65)
        assert le_0 == pytest.approx(le_65, abs=_TOL_DETERMINISTIC)

    def Test_Life_Expectancy_Analytical_Formula(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Life expectancy must equal (1 - qx) / qx for constant model."""
        qx = CONSTANT_DEFAULT_QX
        expected_le = (1.0 - qx) / qx
        actual_le = fitted_constant_default.get_life_expectancy(0)
        assert actual_le == pytest.approx(expected_le, abs=_TOL_DETERMINISTIC)


# ======================================================================
# Constant model — Survival probability
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Constant_Survival:
    """Regression tests for constant-model survival probabilities."""

    def Test_Survival_Matches_Baseline(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Survival probabilities must match the stored baseline."""
        baseline = load_baseline("constant_survival_default")

        for idx_row in range(len(baseline)):
            age = int(baseline["Age"][idx_row])
            t_yr = float(baseline["t_years"][idx_row])
            expected_prob = float(baseline["Survival_Probability"][idx_row])
            actual_prob = fitted_constant_default.survival_probability(age, t_yr)
            assert actual_prob == pytest.approx(expected_prob, abs=_TOL_DETERMINISTIC), (
                f"Survival mismatch at age={age}, t={t_yr}"
            )

    def Test_Survival_Analytical_Formula(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Survival must equal (1 - qx)^t for any age under constant model."""
        qx = CONSTANT_DEFAULT_QX
        for age in SURVIVAL_AGES:
            for t_yr in SURVIVAL_HORIZONS:
                expected = (1.0 - qx) ** t_yr
                actual = fitted_constant_default.survival_probability(age, t_yr)
                assert actual == pytest.approx(expected, abs=_TOL_DETERMINISTIC), (
                    f"Survival formula mismatch at age={age}, t={t_yr}"
                )

    def Test_Survival_Is_Age_Independent(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Under constant mortality, survival probability is independent of age."""
        t_yr = 10.0
        prob_40 = fitted_constant_default.survival_probability(40, t_yr)
        prob_80 = fitted_constant_default.survival_probability(80, t_yr)
        assert prob_40 == pytest.approx(prob_80, abs=_TOL_DETERMINISTIC)

    def Test_Survival_Decreases_With_Horizon(
        self,
        fitted_constant_default: Longevity_Model_Constant,
    ) -> None:
        """Survival probability must decrease as the horizon increases."""
        age = 65
        probs = [
            fitted_constant_default.survival_probability(age, t_yr) for t_yr in SURVIVAL_HORIZONS
        ]
        for idx_t in range(1, len(probs)):
            assert probs[idx_t] < probs[idx_t - 1], (
                f"Survival not decreasing: t={SURVIVAL_HORIZONS[idx_t]}"
            )


# ======================================================================
# Gompertz model — Fitted parameters
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Gompertz_Fitted_Parameters:
    """Regression tests for Gompertz parameter recovery via OLS."""

    def Test_Parameters_Match_Baseline(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Fitted parameters must match the stored baseline."""
        baseline = load_baseline("gompertz_parameters_fitted")
        actual = pl.DataFrame(
            {key: [val] for key, val in fitted_gompertz.parameters.items()},
        )
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_NUMERICAL)

    def Test_B_Is_Positive(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Baseline mortality B must be positive."""
        assert fitted_gompertz.parameters["B"] > 0

    def Test_b_Is_Positive(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Gompertz slope b must be positive (mortality increases with age)."""
        assert fitted_gompertz.parameters["b"] > 0

    def Test_B_Is_Finite(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Both parameters must be finite."""
        assert np.isfinite(fitted_gompertz.parameters["B"])
        assert np.isfinite(fitted_gompertz.parameters["b"])

    def Test_Status_Is_Fitted(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Model status must be FITTED after fit()."""
        assert fitted_gompertz.is_fitted
        assert fitted_gompertz.status == Longevity_Model_Status.FITTED

    def Test_OLS_Recovers_True_Parameters(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Since data was generated with B=5e-5, b=0.09, OLS should recover them closely."""
        assert fitted_gompertz.parameters["B"] == pytest.approx(5e-5, rel=0.01)
        assert fitted_gompertz.parameters["b"] == pytest.approx(0.09, rel=0.01)


# ======================================================================
# Gompertz model — Predictions
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Gompertz_Predictions:
    """Regression tests for Gompertz mortality-table predictions."""

    def Test_Short_Prediction_Matches_Baseline(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """20-age prediction must match the stored baseline."""
        baseline = load_baseline("gompertz_predict_20ages")
        actual = fitted_gompertz.predict(
            n_ages=GOMPERTZ_N_AGES_SHORT,
            start_age=GOMPERTZ_START_AGE,
        )
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_NUMERICAL)

    def Test_Long_Prediction_Matches_Baseline(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """50-age prediction must match the stored baseline."""
        baseline = load_baseline("gompertz_predict_50ages")
        actual = fitted_gompertz.predict(
            n_ages=GOMPERTZ_N_AGES_LONG,
            start_age=GOMPERTZ_START_AGE,
        )
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_NUMERICAL)

    def Test_Summary_Matches_Baseline(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Summary statistics of 20-age prediction must match baseline."""
        baseline = load_baseline("gompertz_summary_20ages")
        pred = fitted_gompertz.predict(
            n_ages=GOMPERTZ_N_AGES_SHORT,
            start_age=GOMPERTZ_START_AGE,
        )
        actual = fitted_gompertz.get_summary_statistics(pred)
        _assert_dataframe_matches_baseline(actual, baseline, _TOL_NUMERICAL)

    def Test_Short_Prediction_Shape(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Short prediction must have correct shape (20, 3): Age, qx, survival."""
        pred = fitted_gompertz.predict(
            n_ages=GOMPERTZ_N_AGES_SHORT,
            start_age=GOMPERTZ_START_AGE,
        )
        assert pred.shape == (GOMPERTZ_N_AGES_SHORT, 3)

    def Test_Long_Prediction_Shape(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Long prediction must have correct shape (50, 3)."""
        pred = fitted_gompertz.predict(
            n_ages=GOMPERTZ_N_AGES_LONG,
            start_age=GOMPERTZ_START_AGE,
        )
        assert pred.shape == (GOMPERTZ_N_AGES_LONG, 3)

    def Test_Qx_Increases_With_Age(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Under Gompertz, qx must be monotonically increasing with age."""
        pred = fitted_gompertz.predict(
            n_ages=GOMPERTZ_N_AGES_SHORT,
            start_age=GOMPERTZ_START_AGE,
        )
        qx_vals = pred["qx"].to_list()
        for idx_age in range(1, len(qx_vals)):
            assert qx_vals[idx_age] > qx_vals[idx_age - 1], (
                f"qx not increasing at age index {idx_age}"
            )

    def Test_Survival_Column_Starts_At_One(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Survival column must start at 1.0 (at start_age)."""
        pred = fitted_gompertz.predict(
            n_ages=GOMPERTZ_N_AGES_SHORT,
            start_age=GOMPERTZ_START_AGE,
        )
        assert pred["survival"][0] == pytest.approx(1.0, abs=_TOL_DETERMINISTIC)

    def Test_Survival_Column_Decreases(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Survival column must be monotonically decreasing."""
        pred = fitted_gompertz.predict(
            n_ages=GOMPERTZ_N_AGES_SHORT,
            start_age=GOMPERTZ_START_AGE,
        )
        surv_vals = pred["survival"].to_list()
        for idx_age in range(1, len(surv_vals)):
            assert surv_vals[idx_age] < surv_vals[idx_age - 1], (
                f"Survival not decreasing at age index {idx_age}"
            )

    def Test_Qx_Between_Zero_And_One(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """All qx values must be in (0, 1)."""
        pred = fitted_gompertz.predict(
            n_ages=GOMPERTZ_N_AGES_SHORT,
            start_age=GOMPERTZ_START_AGE,
        )
        qx_vals = pred["qx"].to_numpy()
        assert np.all(qx_vals > 0.0)
        assert np.all(qx_vals < 1.0)


# ======================================================================
# Gompertz model — Life expectancy
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Gompertz_Life_Expectancy:
    """Regression tests for Gompertz life expectancy at various ages."""

    def Test_Life_Expectancy_Matches_Baseline(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Life expectancy at canonical ages must match the stored baseline."""
        baseline = load_baseline("gompertz_life_expectancy")

        for idx_row in range(len(baseline)):
            age = int(baseline["Age"][idx_row])
            expected_le = float(baseline["Life_Expectancy"][idx_row])
            actual_le = fitted_gompertz.get_life_expectancy(age)
            assert actual_le == pytest.approx(expected_le, abs=_TOL_MODERATE), (
                f"Gompertz life expectancy mismatch at age {age}"
            )

    def Test_Life_Expectancy_Decreases_With_Age(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Remaining life expectancy must decrease as starting age increases."""
        le_values = [fitted_gompertz.get_life_expectancy(age) for age in LIFE_EXPECTANCY_AGES]
        for idx_le in range(1, len(le_values)):
            assert le_values[idx_le] < le_values[idx_le - 1], (
                f"LE not decreasing from age {LIFE_EXPECTANCY_AGES[idx_le - 1]} "
                f"to {LIFE_EXPECTANCY_AGES[idx_le]}"
            )

    def Test_Life_Expectancy_Is_Positive(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Life expectancy must be positive at all tested ages."""
        for age in LIFE_EXPECTANCY_AGES:
            le_val = fitted_gompertz.get_life_expectancy(age)
            assert le_val > 0.0, f"LE not positive at age {age}"

    def Test_Life_Expectancy_At_65_Is_Reasonable(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Gompertz LE at age 65 should be roughly 15-25 years for US-calibrated params."""
        le_65 = fitted_gompertz.get_life_expectancy(65)
        assert 10.0 < le_65 < 30.0, f"Unreasonable LE at 65: {le_65}"


# ======================================================================
# Gompertz model — Survival probability
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Gompertz_Survival:
    """Regression tests for Gompertz analytical survival probabilities."""

    def Test_Survival_Matches_Baseline(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Survival grid must match the stored baseline."""
        baseline = load_baseline("gompertz_survival_probability")

        for idx_row in range(len(baseline)):
            age = int(baseline["Age"][idx_row])
            t_yr = float(baseline["t_years"][idx_row])
            expected_prob = float(baseline["Survival_Probability"][idx_row])
            actual_prob = fitted_gompertz.survival_probability(age, t_yr)
            assert actual_prob == pytest.approx(expected_prob, abs=_TOL_NUMERICAL), (
                f"Gompertz survival mismatch at age={age}, t={t_yr}"
            )

    def Test_Survival_Decreases_With_Horizon(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Survival probability must decrease as the projection horizon increases."""
        age = 65
        probs = [fitted_gompertz.survival_probability(age, t_yr) for t_yr in SURVIVAL_HORIZONS]
        for idx_t in range(1, len(probs)):
            assert probs[idx_t] < probs[idx_t - 1], (
                f"Survival not decreasing from t={SURVIVAL_HORIZONS[idx_t - 1]} "
                f"to t={SURVIVAL_HORIZONS[idx_t]}"
            )

    def Test_Survival_Decreases_With_Age(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """At fixed horizon, survival decreases for older starting ages."""
        t_yr = 10.0
        probs = [fitted_gompertz.survival_probability(age, t_yr) for age in SURVIVAL_AGES]
        for idx_a in range(1, len(probs)):
            assert probs[idx_a] < probs[idx_a - 1], (
                f"Survival not decreasing from age {SURVIVAL_AGES[idx_a - 1]} "
                f"to age {SURVIVAL_AGES[idx_a]}"
            )

    def Test_Survival_Between_Zero_And_One(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """All survival probabilities must be in (0, 1]."""
        for age in SURVIVAL_AGES:
            for t_yr in SURVIVAL_HORIZONS:
                prob = fitted_gompertz.survival_probability(age, t_yr)
                assert 0.0 < prob <= 1.0, f"Survival out of range at age={age}, t={t_yr}: {prob}"

    def Test_Survival_Short_Horizon_Near_One(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """For a very short horizon, survival should be very close to 1."""
        prob = fitted_gompertz.survival_probability(40, 0.01)
        assert prob > 0.99


# ======================================================================
# Gompertz model — Force of mortality
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Gompertz_Force_Of_Mortality:
    """Regression tests for the Gompertz hazard rate at selected ages."""

    def Test_Force_Of_Mortality_Matches_Baseline(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Force of mortality at canonical ages must match baseline."""
        baseline = load_baseline("gompertz_force_of_mortality")

        for idx_row in range(len(baseline)):
            age = int(baseline["Age"][idx_row])
            expected_mu = float(baseline["Force_Of_Mortality"][idx_row])
            actual_mu = fitted_gompertz.get_force_of_mortality(age)
            assert actual_mu == pytest.approx(expected_mu, abs=_TOL_NUMERICAL), (
                f"Force of mortality mismatch at age {age}"
            )

    def Test_Force_Of_Mortality_Increases_With_Age(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Gompertz hazard rate must increase with age."""
        mu_values = [fitted_gompertz.get_force_of_mortality(age) for age in FORCE_OF_MORTALITY_AGES]
        for idx_mu in range(1, len(mu_values)):
            assert mu_values[idx_mu] > mu_values[idx_mu - 1], (
                f"mu not increasing from age {FORCE_OF_MORTALITY_AGES[idx_mu - 1]} "
                f"to {FORCE_OF_MORTALITY_AGES[idx_mu]}"
            )

    def Test_Force_Of_Mortality_Is_Positive(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Hazard rate must be positive at all ages."""
        for age in FORCE_OF_MORTALITY_AGES:
            mu_val = fitted_gompertz.get_force_of_mortality(age)
            assert mu_val > 0.0, f"mu not positive at age {age}"

    def Test_Force_Of_Mortality_Analytical(
        self,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """mu(x) must equal B * exp(b * x) for the fitted parameters."""
        B_fitted = fitted_gompertz.parameters["B"]
        b_fitted = fitted_gompertz.parameters["b"]

        for age in FORCE_OF_MORTALITY_AGES:
            expected_mu = B_fitted * np.exp(b_fitted * age)
            actual_mu = fitted_gompertz.get_force_of_mortality(age)
            assert actual_mu == pytest.approx(expected_mu, abs=_TOL_NUMERICAL), (
                f"Analytical mu mismatch at age {age}"
            )


# ======================================================================
# Gompertz model — Historical data round-trip
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Gompertz_Historical_Data:
    """Verify the canonical Gompertz life-table data matches its baseline."""

    def Test_Historical_Data_Matches_Baseline(
        self,
        gompertz_historical_qx: pl.DataFrame,
    ) -> None:
        """Input data must match the stored Parquet baseline."""
        baseline = load_baseline("gompertz_historical_qx")
        _assert_dataframe_matches_baseline(
            gompertz_historical_qx,
            baseline,
            _TOL_NUMERICAL,
        )

    def Test_Historical_Data_Shape(
        self,
        gompertz_historical_qx: pl.DataFrame,
    ) -> None:
        """Historical data must have 60 rows and 2 columns."""
        assert gompertz_historical_qx.shape == (60, 2)

    def Test_Historical_Data_All_Finite(
        self,
        gompertz_historical_qx: pl.DataFrame,
    ) -> None:
        """All qx values must be finite."""
        qx_arr = gompertz_historical_qx["qx"].to_numpy()
        assert np.all(np.isfinite(qx_arr))


# ======================================================================
# Cross-model consistency
# ======================================================================


@pytest.mark.regression()
class Class_Test_Regression_Cross_Model_Consistency:
    """Sanity checks that hold across both longevity models."""

    def Test_Both_Models_Are_Fitted(
        self,
        fitted_constant_default: Longevity_Model_Constant,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Both models must be in FITTED state."""
        assert fitted_constant_default.is_fitted
        assert fitted_gompertz.is_fitted

    def Test_Predictions_Have_Correct_Column_Schemas(
        self,
        fitted_constant_default: Longevity_Model_Constant,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Both models' predictions must contain 'Age' and 'qx' columns."""
        pred_const = fitted_constant_default.predict(n_ages=5, start_age=65)
        pred_gomp = fitted_gompertz.predict(n_ages=5, start_age=65)

        assert "Age" in pred_const.columns
        assert "qx" in pred_const.columns
        assert "Age" in pred_gomp.columns
        assert "qx" in pred_gomp.columns

    def Test_Gompertz_Qx_At_75_Is_Larger_Than_Constant_Default(
        self,
        fitted_constant_default: Longevity_Model_Constant,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """At age 75, Gompertz qx should exceed the constant default (0.02)."""
        pred_const = fitted_constant_default.predict(n_ages=1, start_age=75)
        pred_gomp = fitted_gompertz.predict(n_ages=1, start_age=75)
        assert pred_gomp["qx"][0] > pred_const["qx"][0]

    def Test_Gompertz_Life_Expectancy_At_65_Differs_From_Constant(
        self,
        fitted_constant_default: Longevity_Model_Constant,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """Life expectancy from the two models must differ (different assumptions)."""
        le_const = fitted_constant_default.get_life_expectancy(65)
        le_gomp = fitted_gompertz.get_life_expectancy(65)
        assert le_const != pytest.approx(le_gomp, abs=1.0)

    def Test_Survival_Ordering_At_High_Age(
        self,
        fitted_constant_default: Longevity_Model_Constant,
        fitted_gompertz: Longevity_Model_Standard,
    ) -> None:
        """At age 80 with t=20, Gompertz should yield lower survival than constant 2%."""
        surv_const = fitted_constant_default.survival_probability(80, 20.0)
        surv_gomp = fitted_gompertz.survival_probability(80, 20.0)
        assert surv_gomp < surv_const
