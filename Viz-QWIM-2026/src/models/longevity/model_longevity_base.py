r"""Abstract base class for QWIM longevity / mortality models.

=========================================================

Provides :class:`Longevity_Model_Base`, an abstract base class that
defines the common interface for all mortality and longevity models used
in the QWIM retirement-planning and actuarial pipeline.

Concrete subclasses implement two primary responsibilities:

1. **Fitting** — estimate (or accept) the model parameters from
   historical mortality data (life tables) or actuarial assumptions.
2. **Prediction** — produce a mortality table projecting annual
   death probabilities across a range of ages as a
   :class:`polars.DataFrame`.

Design goals
------------
* **Consistency** — all models share the same ``fit`` / ``predict``
  interface so they are interchangeable downstream.
* **Actuarial convention** — data is expressed in standard life-table
  form: column ``"Age"`` (integer) and ``"qx"`` (annual probability
  of death in decimal form, e.g. ``0.02`` for 2 %).
* **Safety** — thorough input validation with early returns and
  descriptive exceptions before any arithmetic.
* **Auditability** — every state transition is written to the
  project logger in ``%s`` format (no f-strings in log calls).

Author
------
QWIM Team

Version
-------
1.0.0 (2026-07-01)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from aenum import Enum

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)


# ======================================================================
# Enums
# ======================================================================


class Longevity_Model_Status(Enum):  # noqa: N801  # type: ignore[reportGeneralTypeIssues]
    """Lifecycle status of a longevity model instance.

    Attributes
    ----------
    NOT_FITTED : str
        Model has been constructed but ``fit()`` has not been called.
    FITTED : str
        Model has been successfully fitted to data.
    FAILED : str
        The most recent ``fit()`` or ``predict()`` call raised an error.
    """

    NOT_FITTED = "Not Fitted"
    FITTED = "Fitted"
    FAILED = "Failed"


# ======================================================================
# Abstract base class
# ======================================================================


class Longevity_Model_Base(ABC):  # noqa: N801
    r"""Abstract base class for QWIM longevity and mortality models.

    All concrete longevity models must inherit from this class and
    implement the abstract methods :meth:`fit`, :meth:`predict`,
    :meth:`get_life_expectancy`, and :meth:`survival_probability`.

    Parameters
    ----------
    name_model : str, optional
        Human-readable label for the model
        (default ``"Longevity Model"``).

    Notes
    -----
    Members follow the project convention of ``m_`` prefix for instance
    attributes that back public properties.

    Historical data passed to :meth:`fit` must contain at minimum an
    ``"Age"`` column (non-negative integers) and a ``"qx"`` column
    with annual death probabilities in decimal form
    (e.g. ``0.02`` for 2 %).

    The quantity :math:`q_x` (read: *q-sub-x*) is the actuarial
    probability that a life aged exactly :math:`x` will die before
    reaching age :math:`x + 1`, i.e.

    .. math::

        q_x = 1 - p_x = 1 - \frac{l_{x+1}}{l_x}

    where :math:`l_x` is the expected number of survivors at
    exact age :math:`x` out of :math:`l_0 = 100{,}000` new-borns.

    Examples
    --------
    Subclasses override ``fit`` and ``predict``:

    .. code-block:: python

        class My_Longevity_Model(Longevity_Model_Base):
            def fit(self, data):
                # estimate parameters
                self.m_status = Longevity_Model_Status.FITTED
                return self

            def predict(self, n_ages, start_age=0):
                # produce mortality table
                ...
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name_model: str = "Longevity Model") -> None:
        if not isinstance(name_model, str) or not name_model.strip():
            raise Exception_Validation_Input(
                "name_model must be a non-empty string",
                field_name="name_model",
                expected_type=str,
                actual_value=name_model,
            )

        self.m_name_model: str = name_model.strip()
        self.m_status: Longevity_Model_Status = Longevity_Model_Status.NOT_FITTED  # type: ignore[reportAttributeAccessIssue]
        self.m_parameters: dict[str, float] = {}

        logger.info(
            "Longevity_Model_Base created: '%s'",
            self.m_name_model,
        )

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> Longevity_Model_Base:
        """Fit (calibrate) the model to historical mortality data.

        Concrete subclasses must:

        1. Validate the incoming ``data`` DataFrame.
        2. Estimate model-specific parameters.
        3. Set ``self.m_status = Longevity_Model_Status.FITTED``.
        4. Return ``self`` for method chaining.

        Parameters
        ----------
        data : pl.DataFrame
            Historical mortality data.  If non-empty, must contain at
            minimum an ``"Age"`` column (non-negative integers) and a
            ``"qx"`` column (annual death probabilities in decimal form,
            e.g. ``0.02`` for 2 %).
            Pass an empty ``pl.DataFrame()`` to skip estimation and use
            constructor-supplied parameters directly.

        Returns
        -------
        Longevity_Model_Base
            The fitted model instance (``self``).

        Raises
        ------
        Exception_Validation_Input
            If ``data`` is missing required columns or has invalid values.
        """

    @abstractmethod
    def predict(self, n_ages: int, start_age: int = 0) -> pl.DataFrame:
        """Generate a mortality table for a range of ages.

        Concrete subclasses must:

        1. Verify the model is fitted (``self.m_status == FITTED``).
        2. Produce one projected :math:`q_x` per age.
        3. Return a tidy :class:`polars.DataFrame`.

        Parameters
        ----------
        n_ages : int
            Number of one-year age intervals to project (must be >= 1).
        start_age : int, optional
            Starting age of the projection in whole years (default ``0``).

        Returns
        -------
        pl.DataFrame
            DataFrame with at minimum:

            * ``"Age"`` — :class:`polars.Int64`, age in completed years
            * ``"qx"`` — :class:`polars.Float64`, annual probability of
              death in decimal form

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If ``n_ages`` is not a positive integer or ``start_age``
            is negative.
        """

    @abstractmethod
    def get_life_expectancy(self, current_age: int = 0) -> float:
        r"""Return the expected remaining lifetime at *current_age*.

        The curtate life expectancy (in complete years) from age
        :math:`x_0` is:

        .. math::

            e_{x_0} = \sum_{k=1}^{\omega - x_0}
                       \prod_{j=0}^{k-1} p_{x_0 + j}

        where :math:`p_x = 1 - q_x` and :math:`\omega` is the limiting
        age (default ``120``).

        Parameters
        ----------
        current_age : int, optional
            Age in completed years from which remaining life expectancy
            is computed (default ``0``).

        Returns
        -------
        float
            Expected remaining lifetime in years.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If ``current_age`` is negative.
        """

    @abstractmethod
    def survival_probability(self, current_age: int, t_years: float) -> float:
        r"""Compute the probability of surviving *t_years* from *current_age*.

        Returns :math:`{}_t p_x`, the probability that a life aged
        exactly :math:`x` will survive to age :math:`x + t`.

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
            If ``current_age`` is negative or ``t_years`` is not
            a positive finite number.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name_model(self) -> str:
        """Str : Human-readable model label."""
        return self.m_name_model

    @property
    def status(self) -> Longevity_Model_Status:
        """Longevity_Model_Status : Current lifecycle status."""
        return self.m_status

    @property
    def is_fitted(self) -> bool:
        """Bool : ``True`` iff the model has been fitted successfully."""
        return self.m_status == Longevity_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]

    @property
    def parameters(self) -> dict[str, float]:
        """dict[str, float] : Model parameters (copy)."""
        return self.m_parameters.copy()

    # ------------------------------------------------------------------
    # Shared validation helpers
    # ------------------------------------------------------------------

    def _validate_n_ages(self, n_ages: int) -> None:
        """Validate that *n_ages* is a positive integer.

        Parameters
        ----------
        n_ages : int
            Number of age intervals to validate.

        Raises
        ------
        Exception_Validation_Input
            If ``n_ages`` is not an ``int`` or is less than 1.
        """
        if not isinstance(n_ages, int) or n_ages < 1:
            raise Exception_Validation_Input(
                "n_ages must be a positive integer (>= 1)",
                field_name="n_ages",
                expected_type=int,
                actual_value=n_ages,
            )

    def _validate_age(self, age: int) -> None:
        """Validate that *age* is a non-negative integer.

        Parameters
        ----------
        age : int
            Age value to validate.

        Raises
        ------
        Exception_Validation_Input
            If ``age`` is not an ``int`` or is negative.
        """
        if not isinstance(age, int) or age < 0:
            raise Exception_Validation_Input(
                "age must be a non-negative integer",
                field_name="age",
                expected_type=int,
                actual_value=age,
            )

    def _validate_mortality_data(self, data: pl.DataFrame) -> None:
        """Validate structure and content of historical mortality data.

        Parameters
        ----------
        data : pl.DataFrame
            Input DataFrame to validate.

        Raises
        ------
        Exception_Validation_Input
            If ``data`` is not a :class:`polars.DataFrame`, is empty,
            or is missing required columns (``"Age"``, ``"qx"``).
        """
        if not isinstance(data, pl.DataFrame):
            raise Exception_Validation_Input(
                "data must be a polars.DataFrame",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value=type(data).__name__,
            )

        if len(data) == 0:
            raise Exception_Validation_Input(
                "data must not be empty",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value="empty DataFrame",
            )

        required_cols = {"Age", "qx"}
        missing = required_cols - set(data.columns)
        if missing:
            raise Exception_Validation_Input(
                f"data is missing required columns: {sorted(missing)}",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value=data.columns,
            )

    def _check_is_fitted(self) -> None:
        """Assert the model has been fitted.

        Raises
        ------
        Exception_Calculation
            If :attr:`is_fitted` is ``False``.
        """
        if not self.is_fitted:
            raise Exception_Calculation(
                f"Model '{self.m_name_model}' has not been fitted — call fit() first",
            )

    # ------------------------------------------------------------------
    # Shared utility methods
    # ------------------------------------------------------------------

    def get_summary_statistics(self, df_predict: pl.DataFrame) -> pl.DataFrame:
        """Compute basic summary statistics over a mortality prediction DataFrame.

        Parameters
        ----------
        df_predict : pl.DataFrame
            Output of :meth:`predict` with a ``"qx"`` column.

        Returns
        -------
        pl.DataFrame
            Single-row DataFrame with columns ``"Mean"``, ``"Median"``,
            ``"Std"``, ``"Min"``, ``"Max"``, ``"P5"``, ``"P95"``.

        Raises
        ------
        Exception_Validation_Input
            If ``df_predict`` lacks a ``"qx"`` column.
        """
        if "qx" not in df_predict.columns:
            raise Exception_Validation_Input(
                "df_predict must contain a 'qx' column",
                field_name="df_predict",
                expected_type=pl.DataFrame,
                actual_value=df_predict.columns,
            )

        rates = df_predict["qx"].to_numpy()

        return pl.DataFrame(
            {
                "Mean": [float(np.mean(rates))],
                "Median": [float(np.median(rates))],
                "Std": [float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0],
                "Min": [float(np.min(rates))],
                "Max": [float(np.max(rates))],
                "P5": [float(np.percentile(rates, 5))],
                "P95": [float(np.percentile(rates, 95))],
            },
        )

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Longevity_Model_Base(name='{self.m_name_model}', status={self.m_status.value})"
