r"""Constant mortality-rate model for the QWIM actuarial pipeline.

=========================================================

Provides :class:`Longevity_Model_Constant`, a deterministic model that
assumes a **constant annual probability of death** :math:`q` for every
age.  Under this exponential-lifetime assumption the force of mortality
is constant:

.. math::

    \mu = -\ln(1 - q)

which implies:

* **Survival function**: :math:`S(t) = (1-q)^t = e^{-\mu t}`
* **Remaining life expectancy**: :math:`e_x = \tfrac{1 - q}{q} \approx \tfrac{1}{\mu}`
  (geometric-series formula, age-independent)

This is the simplest possible mortality assumption and is used as a
baseline or sensitivity anchor in retirement-income, annuity-pricing,
and liability-valuation models.  When historical life-table data is
supplied to :meth:`fit`, the model estimates :math:`q` as the
arithmetic mean of the observed :math:`q_x` column; otherwise it uses
the constructor-supplied value directly.

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

_DEFAULT_QX: float = 0.01  # 1 % annual death probability (≈ age 50 US male)

# Valid range for a constant annual death probability
_QX_MIN: float = 1e-6  # practically zero (infants / very healthy populations)
_QX_MAX: float = 0.5  # 50 % — boundary for individually extreme scenarios


# ======================================================================
# Constant mortality-rate model
# ======================================================================


class Longevity_Model_Constant(Longevity_Model_Base):  # noqa: N801
    r"""Deterministic constant-mortality model (exponential lifetime).

    Projects the same annual death probability :math:`q` for every
    future age:

    .. math::

        q_x = q \quad \forall\; x \geq 0

    Under this exponential-lifetime assumption:

    * **Survival probability**: :math:`{}_t p_x = (1-q)^t`
    * **Remaining life expectancy**: :math:`e_x = \tfrac{1-q}{q}`

    Parameters
    ----------
    qx : float, optional
        Fixed annual probability of death in decimal form
        (default ``0.01`` for 1 %).  Used when ``fit()`` is called
        without data or when acting as a hard-coded mortality assumption.
    name_model : str, optional
        Human-readable label (default ``"Constant Mortality"``).

    Examples
    --------
    Hard-coded rate, no fitting required:

    >>> model = Longevity_Model_Constant(qx=0.02)
    >>> model.fit(pl.DataFrame())  # no-op fit
    Longevity_Model_Constant(name='Constant Mortality', qx=0.020000, status=Fitted)
    >>> df = model.predict(n_ages=3, start_age=65)
    >>> df["qx"].to_list()
    [0.02, 0.02, 0.02]

    Fit to historical data:

    >>> import polars as pl
    >>> life_table = pl.DataFrame(
    ...     {
    ...         "Age": [60, 65, 70, 75],
    ...         "qx": [0.010, 0.014, 0.021, 0.032],
    ...     }
    ... )
    >>> model = Longevity_Model_Constant()
    >>> model.fit(life_table)
    Longevity_Model_Constant(name='Constant Mortality', qx=0.019250, status=Fitted)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        qx: float = _DEFAULT_QX,
        name_model: str = "Constant Mortality",
    ) -> None:
        super().__init__(name_model=name_model)

        if not isinstance(qx, float | int) or not np.isfinite(qx):
            raise Exception_Validation_Input(
                "qx must be a finite number",
                field_name="qx",
                expected_type=float,
                actual_value=qx,
            )
        if qx < _QX_MIN or qx > _QX_MAX:
            raise Exception_Validation_Input(
                f"qx must be in the range [{_QX_MIN}, {_QX_MAX}]",
                field_name="qx",
                expected_type=float,
                actual_value=qx,
            )

        self.m_qx: float = float(qx)

        logger.info(
            "Longevity_Model_Constant constructed with qx=%.6f",
            self.m_qx,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> Longevity_Model_Constant:
        """Fit the constant model to historical mortality data.

        If *data* is a non-empty DataFrame with the required columns, the
        model updates :attr:`m_qx` to the arithmetic mean of the
        observed ``"qx"`` column.  If an empty DataFrame is passed the
        constructor value is kept unchanged.

        Parameters
        ----------
        data : pl.DataFrame
            Historical mortality data.  If non-empty, must contain
            columns ``"Age"`` and ``"qx"`` (annual death probabilities,
            decimal).  Pass an empty ``pl.DataFrame()`` to skip
            estimation and use the constructor-supplied rate.

        Returns
        -------
        Longevity_Model_Constant
            The fitted model instance (``self``).

        Raises
        ------
        Exception_Validation_Input
            If *data* is not a :class:`polars.DataFrame`, or if it is
            non-empty and missing required columns or contains invalid
            (non-finite) values.
        """
        if not isinstance(data, pl.DataFrame):
            raise Exception_Validation_Input(
                "data must be a polars.DataFrame",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value=type(data).__name__,
            )

        if len(data) > 0:
            self._validate_mortality_data(data)

            qx_values = data["qx"].to_numpy()

            if not np.all(np.isfinite(qx_values)):
                raise Exception_Validation_Input(
                    "qx column contains non-finite values",
                    field_name="qx",
                    expected_type=float,
                    actual_value="non-finite entry",
                )

            self.m_qx = float(np.mean(qx_values))
            logger.info(
                "Longevity_Model_Constant fitted from %d observations; estimated qx=%.6f",
                len(qx_values),
                self.m_qx,
            )
        else:
            logger.info(
                "Longevity_Model_Constant: empty data — using constructor qx=%.6f",
                self.m_qx,
            )

        self.m_status = Longevity_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]
        self.m_parameters = {"qx": self.m_qx}
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, n_ages: int, start_age: int = 0) -> pl.DataFrame:
        r"""Project a constant :math:`q_x` table over *n_ages* age intervals.

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
            * ``"qx"`` — :class:`polars.Float64`

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *n_ages* is not a positive integer or *start_age*
            is negative.

        Examples
        --------
        >>> model = Longevity_Model_Constant(qx=0.02)
        >>> model.fit(pl.DataFrame())
        >>> df = model.predict(n_ages=4, start_age=65)
        >>> len(df)
        4
        >>> df["qx"].to_list()
        [0.02, 0.02, 0.02, 0.02]
        """
        self._check_is_fitted()
        self._validate_n_ages(n_ages)
        self._validate_age(start_age)

        ages = list(range(start_age, start_age + n_ages))

        logger.info(
            "Longevity_Model_Constant.predict: n_ages=%d, start_age=%d, qx=%.6f",
            n_ages,
            start_age,
            self.m_qx,
        )

        return pl.DataFrame(
            {
                "Age": ages,
                "qx": [self.m_qx] * n_ages,
            },
        )

    # ------------------------------------------------------------------
    # Life expectancy and survival
    # ------------------------------------------------------------------

    def get_life_expectancy(self, current_age: int = 0) -> float:
        r"""Return expected remaining lifetime under constant mortality.

        Under the constant-:math:`q` exponential-lifetime assumption the
        curtate life expectancy is the exact geometric-series result:

        .. math::

            e_x = \frac{1 - q}{q}

        which is **age-independent** because the hazard rate is constant.

        Parameters
        ----------
        current_age : int, optional
            Starting age — ignored for the constant model but validated
            for consistency with the base-class interface (default ``0``).

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

        life_expectancy = (1.0 - self.m_qx) / self.m_qx
        logger.info(
            "Longevity_Model_Constant.get_life_expectancy: age=%d, e_x=%.2f yr",
            current_age,
            life_expectancy,
        )
        return life_expectancy

    def survival_probability(self, current_age: int, t_years: float) -> float:
        r"""Compute the survival probability under constant mortality.

        Under the exponential-lifetime assumption:

        .. math::

            {}_t p_x = (1 - q)^t

        Parameters
        ----------
        current_age : int
            Current age (validated but not used in computation for the
            constant model).
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

        prob = float((1.0 - self.m_qx) ** t_years)
        logger.info(
            "Longevity_Model_Constant.survival_probability: age=%d, t=%.2f yr, prob=%.6f",
            current_age,
            float(t_years),
            prob,
        )
        return prob

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Longevity_Model_Constant("
            f"name='{self.m_name_model}', "
            f"qx={self.m_qx:.6f}, "
            f"status={self.m_status.value})"
        )
