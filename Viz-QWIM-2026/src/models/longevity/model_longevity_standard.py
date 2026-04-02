r"""Standard Gompertz mortality model for the QWIM actuarial pipeline.

=========================================================

Provides :class:`Longevity_Model_Standard`, a deterministic mortality
model based on the **Gompertz** law of mortality — the canonical
single-cohort mortality model in the actuarial and longevity literature.

The Gompertz model specifies the **force of mortality** (hazard rate) as
an exponential function of age:

.. math::

    \mu(x) = B \, e^{\,b\,x}

where :math:`B > 0` is the baseline mortality intensity and
:math:`b > 0` is the rate at which mortality accelerates with age
(the Gompertz slope).

From the force of mortality, the standard actuarial quantities follow:

* **Annual death probability**:

  .. math::

      q_x = 1 - \exp\!\left(
          -\frac{B}{b}\, e^{\,b\,x}\!\left(e^{\,b} - 1\right)
      \right)

* **Survival probability** (:math:`t`-year, continuous):

  .. math::

      {}_t p_x = \exp\!\left(
          -\frac{B}{b}\, e^{\,b\,x}\!\left(e^{\,b\,t} - 1\right)
      \right)

* **Remaining life expectancy** (curtate, discrete sum):

  .. math::

      e_x = \sum_{k=1}^{\omega - x} {}_k p_x

  where :math:`\omega = 120` is the limiting age.

Fitting
-------
Parameters are estimated from a historical life-table by applying the
**complementary log-log (CLL) transform** to :math:`q_x` values, which
linearises the Gompertz relationship:

.. math::

    \ln\!\bigl(-\ln(1 - q_x)\bigr)
    = \underbrace{\ln\!\left(\tfrac{B}{b}(e^b - 1)\right)}_{\alpha}
      + b\, x

OLS on this transformed equation yields simultaneous estimates of
:math:`b` (slope) and :math:`B` (recovered from the intercept).

If the OLS slope is non-positive (mortality not increasing with age),
the model falls back to the constructor prior values.

References
----------
* Gompertz, B. (1825). *On the nature of the function expressive of the
  law of human mortality.* Philosophical Transactions of the Royal
  Society, 115, 513-583.
* Makeham, W.M. (1860). *On the law of mortality and the construction
  of annuity tables.* Journal of the Institute of Actuaries, 8, 301-310.
* Dickson, D.C.M., Hardy, M.R., & Waters, H.R. (2020).
  *Actuarial Mathematics for Life Contingent Risks* (3rd ed.).
  Cambridge University Press.
* Carriere, J.F. (1992). *Parametric models for life tables.*
  Transactions of the Society of Actuaries, 44, 77-99.

Author
------
QWIM Team

Version
-------
1.0.0 (2026-07-01)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from numpy.linalg import lstsq

from src.models.longevity.model_longevity_base import (
    Longevity_Model_Base,
    Longevity_Model_Status,
)
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)

# ======================================================================
# Module-level defaults
# ======================================================================

# Gompertz parameters calibrated to approximate US adult mortality
# (SSA Period Life Table, 2019):
#   mu(65) ≈ 0.017 → B * exp(0.09 * 65) ≈ 0.017 → B ≈ 5e-5
_DEFAULT_B: float = 5e-5  # baseline mortality at age 0
_DEFAULT_b: float = 0.09  # mortality-doubling rate (yr^-1); ~7.7-yr doubling
_MAX_AGE: int = 120  # limiting age for life-expectancy summation


# ======================================================================
# Gompertz mortality model
# ======================================================================


class Longevity_Model_Standard(Longevity_Model_Base):  # noqa: N801
    r"""Gompertz parametric mortality model.

    Implements the exponential force-of-mortality law:

    .. math::

        \mu(x) = B \, e^{\,b\,x}

    Parameters
    ----------
    B : float, optional
        Initial guess for the baseline mortality intensity :math:`B > 0`
        (default ``5e-5``).  Overwritten by :meth:`fit`.
    b : float, optional
        Initial guess for the Gompertz slope :math:`b > 0`
        (default ``0.09``, implying mortality doubles every ≈ 7.7 years).
        Overwritten by :meth:`fit`.
    name_model : str, optional
        Human-readable label (default ``"Gompertz Mortality"``).

    Attributes
    ----------
    m_B : float
        Fitted baseline mortality intensity.
    m_b : float
        Fitted Gompertz slope parameter.

    Notes
    -----
    The Gompertz model is accurate for adult ages (approx. 30-110).  It
    over-estimates infant/child mortality (:math:`x < 30`) and
    under-estimates old-age deceleration (:math:`x > 100`) relative to
    empirical data (the so-called "mortality plateau" phenomenon).
    For those ranges, the Makeham extension (:math:`\mu(x) = A + Be^{bx}`)
    or logistic models may be preferred.

    Examples
    --------
    >>> import polars as pl
    >>> life_table = pl.DataFrame(
    ...     {
    ...         "Age": [40, 45, 50, 55, 60, 65, 70, 75, 80],
    ...         "qx": [0.0031, 0.0049, 0.0077, 0.0119, 0.0181, 0.0270, 0.0401, 0.0594, 0.0877],
    ...     }
    ... )
    >>> model = Longevity_Model_Standard()
    >>> model.fit(life_table)  # doctest: +SKIP
    >>> df = model.predict(n_ages=10, start_age=65)
    >>> "qx" in df.columns
    True
    >>> "survival" in df.columns
    True
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        B: float = _DEFAULT_B,  # noqa: N803
        b: float = _DEFAULT_b,
        name_model: str = "Gompertz Mortality",
    ) -> None:
        super().__init__(name_model=name_model)

        _validate_positive_float(B, "B")
        _validate_positive_float(b, "b")

        self.m_B: float = float(B)
        self.m_b: float = float(b)

        logger.info(
            "Longevity_Model_Standard constructed: B=%.2e, b=%.4f",
            self.m_B,
            self.m_b,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> Longevity_Model_Standard:
        r"""Estimate Gompertz parameters from historical life-table data via OLS.

        Applies the **complementary log-log (CLL) transform**:

        .. math::

            y_x = \ln(-\ln(1 - q_x))
            = \underbrace{\ln\!\left(\tfrac{B}{b}(e^b - 1)\right)}_{\alpha}
              + b\, x

        which is linear in age.  OLS on this equation yields:

        * :math:`\hat{b}` — Gompertz slope (must be > 0)
        * :math:`\hat{B} = \exp(\hat\alpha) \cdot \hat{b} / (e^{\hat{b}} - 1)`

        If :math:`\hat{b} \le 0` (mortality not increasing with age),
        the model falls back to the constructor prior values.

        Parameters
        ----------
        data : pl.DataFrame
            Historical mortality data with columns ``"Age"``
            (non-negative integers) and ``"qx"``
            (annual death probabilities, decimal, must be in ``(0, 1)``).
            Must contain at least 3 rows with ``qx`` strictly between 0
            and 1.  Rows with :math:`q_x = 1` (e.g. limiting-age entry)
            are silently excluded from the OLS fit.

        Returns
        -------
        Longevity_Model_Standard
            The fitted model instance (``self``).

        Raises
        ------
        Exception_Validation_Input
            If *data* is invalid, contains non-finite values, or fewer
            than 3 usable rows (with ``0 < qx < 1``) remain after
            filtering.
        """
        self._validate_mortality_data(data)

        # Sort by age
        data = data.sort("Age")

        ages = data["Age"].to_numpy().astype(np.float64)
        qx_values = data["qx"].to_numpy().astype(np.float64)

        if not np.all(np.isfinite(qx_values)):
            raise Exception_Validation_Input(
                "qx column contains non-finite values",
                field_name="qx",
                expected_type=float,
                actual_value="non-finite entry",
            )

        # Filter out boundary values (qx = 0 or qx = 1) for CLL transform
        mask = (qx_values > 0.0) & (qx_values < 1.0)
        ages_fit = ages[mask]
        qx_fit = qx_values[mask]

        if len(ages_fit) < 3:
            raise Exception_Validation_Input(
                "data must have at least 3 rows with 0 < qx < 1 to fit Gompertz parameters",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value=len(ages_fit),
            )

        # CLL transform: y = ln(-ln(1 - qx)) = alpha + b * age
        y = np.log(-np.log(1.0 - qx_fit))
        x = ages_fit

        a_mat = np.column_stack([np.ones(len(x)), x])
        coeffs, _residuals, *_ = lstsq(a_mat, y, rcond=None)
        alpha_ols, b_ols = coeffs[0], coeffs[1]

        if b_ols <= 0:
            logger.warning(
                "OLS slope b=%.4f <= 0; Gompertz mortality not increasing with age. "
                "Falling back to constructor priors.",
                b_ols,
            )
            b_fitted = self.m_b
            B_fitted = self.m_B
        else:
            b_fitted = float(b_ols)
            # Recover B: exp(alpha) = B/b * (exp(b) - 1)
            # => B = exp(alpha) * b / (exp(b) - 1)
            exp_b = np.exp(b_fitted)
            denom = exp_b - 1.0
            if denom < 1e-12:
                # Limit b → 0: (exp(b)-1)/b → 1, so B ≈ exp(alpha)
                B_fitted = float(np.exp(alpha_ols))
            else:
                B_fitted = float(np.exp(alpha_ols) * b_fitted / denom)

        if not (np.isfinite(b_fitted) and np.isfinite(B_fitted) and B_fitted > 0):
            raise Exception_Validation_Input(
                "OLS produced non-finite or non-positive parameter estimates; check data quality",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value="non-finite OLS result",
            )

        self.m_B = float(B_fitted)
        self.m_b = float(b_fitted)
        self.m_status = Longevity_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]

        self.m_parameters = {
            "B": self.m_B,
            "b": self.m_b,
        }

        logger.info(
            "Longevity_Model_Standard fitted: B=%.2e, b=%.4f (n=%d usable ages)",
            self.m_B,
            self.m_b,
            len(ages_fit),
        )

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, n_ages: int, start_age: int = 0) -> pl.DataFrame:
        r"""Generate the Gompertz mortality table for *n_ages* age intervals.

        For each age :math:`x` in ``[start_age, start_age + n_ages)``
        computes:

        * :math:`q_x` — annual death probability
        * :math:`{}_k p_{start\_age}` — cumulative survival from
          *start_age* to age :math:`x`

        Parameters
        ----------
        n_ages : int
            Number of one-year age intervals to project (must be >= 1).
        start_age : int, optional
            Starting age in completed years (default ``0``).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            * ``"Age"`` — :class:`polars.Int64`
            * ``"qx"`` — :class:`polars.Float64`, annual death probability
            * ``"survival"`` — :class:`polars.Float64`, cumulative survival
              probability from *start_age* (``1.0`` at *start_age*,
              decreasing thereafter)

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *n_ages* is not a positive integer or *start_age*
            is negative.

        Examples
        --------
        >>> model = Longevity_Model_Standard()
        >>> model.fit(life_table)  # doctest: +SKIP
        >>> df = model.predict(n_ages=10, start_age=65)
        >>> len(df)
        10
        >>> df["survival"][0]
        1.0
        """
        self._check_is_fitted()
        self._validate_n_ages(n_ages)
        self._validate_age(start_age)

        ages = list(range(start_age, start_age + n_ages))
        qx_col: list[float] = []
        survival_col: list[float] = []

        cumulative_survival = 1.0
        for age in ages:
            q = self._qx_at_age(age)
            survival_col.append(cumulative_survival)
            qx_col.append(q)
            # Update for next period: cumulative survival decreases by (1-q)
            cumulative_survival *= 1.0 - q

        logger.info(
            "Longevity_Model_Standard.predict: n_ages=%d, start_age=%d",
            n_ages,
            start_age,
        )

        return pl.DataFrame(
            {
                "Age": ages,
                "qx": qx_col,
                "survival": survival_col,
            },
        )

    # ------------------------------------------------------------------
    # Life expectancy and survival
    # ------------------------------------------------------------------

    def get_life_expectancy(self, current_age: int = 0) -> float:
        r"""Compute remaining curtate life expectancy at *current_age*.

        Uses the discrete summation:

        .. math::

            e_{x_0} = \sum_{k=1}^{\omega - x_0} {}_k p_{x_0}

        where :math:`{}_k p_{x_0}` is computed by multiplying
        :math:`k` annual survival factors from the Gompertz model and
        :math:`\omega = 120` is the limiting age.

        Parameters
        ----------
        current_age : int, optional
            Starting age in completed years (default ``0``).

        Returns
        -------
        float
            Expected remaining lifetime in years.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *current_age* is negative.
        """
        self._check_is_fitted()
        self._validate_age(current_age)

        survival = 1.0
        ex = 0.0
        for age in range(current_age, _MAX_AGE):
            q = self._qx_at_age(age)
            survival *= 1.0 - min(q, 1.0)
            ex += survival

        logger.info(
            "Longevity_Model_Standard.get_life_expectancy: age=%d, e_x=%.2f yr",
            current_age,
            ex,
        )
        return ex

    def survival_probability(self, current_age: int, t_years: float) -> float:
        r"""Compute analytical Gompertz survival probability.

        Uses the closed-form expression:

        .. math::

            {}_t p_x = \exp\!\left(
                -\frac{B}{b}\, e^{\,b\,x}\!(e^{\,b\,t} - 1)
            \right)

        Parameters
        ----------
        current_age : int
            Current age in completed years (must be >= 0).
        t_years : float
            Projection horizon in years (must be > 0).

        Returns
        -------
        float
            Survival probability in :math:`[0,\, 1]`.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *current_age* is negative or *t_years* is not a positive
            finite number.
        """
        self._check_is_fitted()
        self._validate_age(current_age)

        if not isinstance(t_years, float | int) or not np.isfinite(t_years) or t_years <= 0:
            raise Exception_Validation_Input(
                "t_years must be a positive finite number",
                field_name="t_years",
                expected_type=float,
                actual_value=t_years,
            )

        t = float(t_years)
        x = float(current_age)
        integral_mu = (self.m_B / self.m_b) * np.exp(self.m_b * x) * (np.exp(self.m_b * t) - 1.0)
        prob = float(np.exp(-integral_mu))

        logger.info(
            "Longevity_Model_Standard.survival_probability: age=%d, t=%.2f yr, prob=%.6f",
            current_age,
            t,
            prob,
        )
        return prob

    # ------------------------------------------------------------------
    # Analytical helpers
    # ------------------------------------------------------------------

    def get_force_of_mortality(self, age: int | float) -> float:
        r"""Compute the Gompertz force of mortality :math:`\mu(x)`.

        .. math::

            \mu(x) = B\, e^{\,b\,x}

        Parameters
        ----------
        age : int or float
            Age at which to evaluate the hazard rate.

        Returns
        -------
        float
            Force of mortality at the given age.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *age* is negative.
        """
        self._check_is_fitted()
        if not isinstance(age, float | int) or age < 0:
            raise Exception_Validation_Input(
                "age must be a non-negative number",
                field_name="age",
                expected_type=float,
                actual_value=age,
            )
        return float(self.m_B * np.exp(self.m_b * float(age)))

    # ------------------------------------------------------------------
    # Private instance helpers
    # ------------------------------------------------------------------

    def _qx_at_age(self, age: float) -> float:
        r"""Compute the Gompertz annual death probability at *age*.

        .. math::

            q_x = 1 - \exp\!\left(
                -\frac{B}{b}\, e^{\,b\,x}\!\left(e^b - 1\right)
            \right)

        Parameters
        ----------
        age : float
            Age at which to evaluate.

        Returns
        -------
        float
            Annual death probability in :math:`[0,\, 1)`.
        """
        integral_mu = (self.m_B / self.m_b) * np.exp(self.m_b * age) * (np.exp(self.m_b) - 1.0)
        return float(1.0 - np.exp(-integral_mu))

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Longevity_Model_Standard("
            f"name='{self.m_name_model}', "
            f"B={self.m_B:.2e}, "
            f"b={self.m_b:.4f}, "
            f"status={self.m_status.value})"
        )


# ======================================================================
# Private helpers
# ======================================================================


def _validate_positive_float(value: float, name: str) -> None:
    """Raise if *value* is not a positive finite float.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Parameter name for the error message.

    Raises
    ------
    Exception_Validation_Input
        If *value* is not a positive finite number.
    """
    if not isinstance(value, float | int) or not np.isfinite(value) or value <= 0:
        raise Exception_Validation_Input(
            f"{name} must be a positive finite number, got {value!r}",
            field_name=name,
            expected_type=float,
            actual_value=value,
        )
