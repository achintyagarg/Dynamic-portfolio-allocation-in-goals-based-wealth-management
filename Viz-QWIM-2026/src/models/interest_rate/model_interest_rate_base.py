r"""Abstract base class for QWIM interest-rate models.

========================================================

Provides :class:`Interest_Rate_Model_Base`, an abstract base class that
defines the common interface for all short-rate models used in the QWIM
retirement-planning and fixed-income pipeline.

Concrete subclasses implement two primary responsibilities:

1. **Fitting** — estimate (or accept) the model parameters from
   historical short-rate time-series data.
2. **Prediction** — produce a time-series of projected short rates
   as a :class:`polars.DataFrame`.

Design goals
------------
* **Consistency** — all models share the same ``fit`` / ``predict``
  interface so they are interchangeable downstream.
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


class Interest_Rate_Model_Status(Enum):  # noqa: N801  # type: ignore[reportGeneralTypeIssues]
    """Lifecycle status of an interest-rate model instance.

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


class Interest_Rate_Model_Base(ABC):  # noqa: N801
    r"""Abstract base class for QWIM short-rate models.

    All concrete interest-rate models must inherit from this class and
    implement the abstract methods :meth:`fit`, :meth:`predict`, and
    :meth:`get_annual_rate`.

    Parameters
    ----------
    name_model : str, optional
        Human-readable label for the model
        (default ``"Interest Rate Model"``).

    Notes
    -----
    Members follow the project convention of ``m_`` prefix for instance
    attributes that back public properties.

    Historical data passed to :meth:`fit` must contain at minimum a
    ``"Date"`` column and a ``"short_rate"`` column with annual rates
    expressed in decimal form (e.g. ``0.05`` for 5 %).

    Examples
    --------
    Subclasses override ``fit`` and ``predict``:

    .. code-block:: python

        class My_Rate_Model(Interest_Rate_Model_Base):
            def fit(self, data):
                # estimate parameters
                self.m_status = Interest_Rate_Model_Status.FITTED
                return self

            def predict(self, n_periods, start_date):
                # produce forecast DataFrame
                ...
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name_model: str = "Interest Rate Model") -> None:
        if not isinstance(name_model, str) or not name_model.strip():
            raise Exception_Validation_Input(
                "name_model must be a non-empty string",
                field_name="name_model",
                expected_type=str,
                actual_value=name_model,
            )

        self.m_name_model: str = name_model.strip()
        self.m_status: Interest_Rate_Model_Status = Interest_Rate_Model_Status.NOT_FITTED  # type: ignore[reportAttributeAccessIssue]
        self.m_parameters: dict[str, float] = {}

        logger.info(
            "Interest_Rate_Model_Base created: '%s'",
            self.m_name_model,
        )

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> Interest_Rate_Model_Base:
        """Fit (calibrate) the model to historical short-rate data.

        Concrete subclasses must:

        1. Validate the incoming ``data`` DataFrame.
        2. Estimate model-specific parameters.
        3. Set ``self.m_status = Interest_Rate_Model_Status.FITTED``.
        4. Return ``self`` for method chaining.

        Parameters
        ----------
        data : pl.DataFrame
            Historical data.  If non-empty, must contain at minimum a
            ``"Date"`` column (:class:`polars.Date`) and a
            ``"short_rate"`` column (:class:`polars.Float64`)
            expressed as annual rates in decimal form
            (e.g. ``0.05`` for 5 %).

        Returns
        -------
        Interest_Rate_Model_Base
            The fitted model instance (``self``).

        Raises
        ------
        Exception_Validation_Input
            If ``data`` is missing required columns or has invalid values.
        """

    @abstractmethod
    def predict(
        self,
        n_periods: int,
        start_date: str | None = None,
        freq: str = "1mo",
    ) -> pl.DataFrame:
        """Generate a forward projection of short rates.

        Concrete subclasses must:

        1. Verify the model is fitted (``self.m_status == FITTED``).
        2. Produce one projected rate per period.
        3. Return a tidy :class:`polars.DataFrame`.

        Parameters
        ----------
        n_periods : int
            Number of periods to project (must be >= 1).
        start_date : str or None, optional
            ISO-8601 start date of the projection window
            (e.g. ``"2025-01-01"``).  If ``None``, defaults to
            ``"2025-01-01"``.
        freq : str, optional
            Polars duration string for the period step
            (default ``"1mo"`` for monthly).

        Returns
        -------
        pl.DataFrame
            DataFrame with at minimum:

            * ``"Date"`` — :class:`polars.Date`
            * ``"short_rate"`` — :class:`polars.Float64`, decimal form

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If ``n_periods`` is not a positive integer.
        """

    @abstractmethod
    def get_annual_rate(self) -> float:
        r"""Return a scalar representative annual short rate.

        For deterministic models this is the fixed rate; for stochastic
        models this is typically the long-run mean (e.g. :math:`\\theta`
        in a Vasicek model).

        Returns
        -------
        float
            Annual short rate in decimal form (e.g. ``0.05``).

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name_model(self) -> str:
        """Str : Human-readable model label."""
        return self.m_name_model

    @property
    def status(self) -> Interest_Rate_Model_Status:
        """Interest_Rate_Model_Status : Current lifecycle status."""
        return self.m_status

    @property
    def is_fitted(self) -> bool:
        """Bool : ``True`` iff the model has been fitted successfully."""
        return self.m_status == Interest_Rate_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]

    @property
    def parameters(self) -> dict[str, float]:
        """dict[str, float] : Model parameters (copy)."""
        return self.m_parameters.copy()

    # ------------------------------------------------------------------
    # Shared validation helpers
    # ------------------------------------------------------------------

    def _validate_n_periods(self, n_periods: int) -> None:
        """Validate that *n_periods* is a positive integer.

        Parameters
        ----------
        n_periods : int
            Number of projection periods to validate.

        Raises
        ------
        Exception_Validation_Input
            If ``n_periods`` is not an ``int`` or is less than 1.
        """
        if not isinstance(n_periods, int) or n_periods < 1:
            raise Exception_Validation_Input(
                "n_periods must be a positive integer (>= 1)",
                field_name="n_periods",
                expected_type=int,
                actual_value=n_periods,
            )

    def _validate_rate_data(self, data: pl.DataFrame) -> None:
        """Validate structure and content of historical short-rate data.

        Parameters
        ----------
        data : pl.DataFrame
            Input DataFrame to validate.

        Raises
        ------
        Exception_Validation_Input
            If ``data`` is not a :class:`polars.DataFrame`, is empty,
            or is missing required columns (``"Date"``, ``"short_rate"``).
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

        required_cols = {"Date", "short_rate"}
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
        """Compute basic summary statistics over a prediction DataFrame.

        Parameters
        ----------
        df_predict : pl.DataFrame
            Output of :meth:`predict` with a ``"short_rate"`` column.

        Returns
        -------
        pl.DataFrame
            Single-row DataFrame with columns ``"Mean"``, ``"Median"``,
            ``"Std"``, ``"Min"``, ``"Max"``, ``"P5"``, ``"P95"``.

        Raises
        ------
        Exception_Validation_Input
            If ``df_predict`` lacks a ``"short_rate"`` column.
        """
        if "short_rate" not in df_predict.columns:
            raise Exception_Validation_Input(
                "df_predict must contain a 'short_rate' column",
                field_name="df_predict",
                expected_type=pl.DataFrame,
                actual_value=df_predict.columns,
            )

        rates = df_predict["short_rate"].to_numpy()

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
        return f"Interest_Rate_Model_Base(name='{self.m_name_model}', status={self.m_status.value})"
