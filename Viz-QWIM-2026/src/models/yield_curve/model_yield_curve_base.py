r"""Abstract base class for QWIM yield curve models.

=========================================================

Provides :class:`Yield_Curve_Model_Base`, an abstract base class that
defines the common interface for all yield curve models used in the QWIM
fixed-income and retirement-planning pipeline.

A *yield curve* (also called the *term structure of interest rates*)
represents the relationship between the spot yield — the zero-coupon
rate expressed as a continuously-compounded or annually-compounded
decimal — and the time to maturity :math:`\tau` for a fixed-income
instrument of a given credit quality and currency.  Examples include
US Treasury benchmark rates, euro-area OIS rates, and investment-grade
corporate-bond benchmark curves.

Concrete subclasses implement two primary responsibilities:

1. **Fitting** — accept observed ``(maturity, yield)`` pairs and
   estimate the model parameters, either by direct assignment
   (constant / flat model) or by numerical regression (parametric
   models such as Nelson-Siegel).
2. **Prediction** — evaluate the fitted curve at an arbitrary set of
   maturities and return a tidy :class:`polars.DataFrame`.

Design goals
------------
* **Consistency** — all models share the same ``fit`` / ``predict``
  interface so they are interchangeable downstream.
* **Convention** — maturities in years (float > 0), yields as decimals
  (e.g. ``0.05`` for 5 %).
* **Safety** — thorough input validation with early returns and
  descriptive exceptions before any arithmetic.
* **Auditability** — every state transition is written to the project
  logger in ``%s`` format (no f-strings in log calls).

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


class Yield_Curve_Model_Status(Enum):  # noqa: N801  # type: ignore[reportGeneralTypeIssues]
    """Lifecycle status of a yield curve model instance.

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


class Yield_Curve_Model_Base(ABC):  # noqa: N801
    r"""Abstract base class for QWIM yield curve models.

    All concrete yield curve models must inherit from this class and
    implement the abstract methods :meth:`fit`, :meth:`predict`,
    :meth:`get_par_yield`, and :meth:`get_forward_rate`.

    Parameters
    ----------
    name_model : str, optional
        Human-readable label for the model
        (default ``"Yield Curve Model"``).

    Notes
    -----
    Members follow the project convention of ``m_`` prefix for instance
    attributes that back public properties.

    Historical data passed to :meth:`fit` must contain at minimum a
    ``"Maturity"`` column (positive float, time to maturity in years) and
    a ``"Yield"`` column with spot yields in decimal form
    (e.g. ``0.05`` for 5 %).

    Examples
    --------
    Subclasses override ``fit`` and ``predict``:

    .. code-block:: python

        class My_Yield_Curve_Model(Yield_Curve_Model_Base):
            def fit(self, data):
                # estimate parameters
                self.m_status = Yield_Curve_Model_Status.FITTED
                return self

            def predict(self, maturities=None, *, n_points=50, max_maturity=30.0):
                # produce yield curve
                ...
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name_model: str = "Yield Curve Model") -> None:
        if not isinstance(name_model, str) or not name_model.strip():
            raise Exception_Validation_Input(
                "name_model must be a non-empty string",
                field_name="name_model",
                expected_type=str,
                actual_value=name_model,
            )

        self.m_name_model: str = name_model.strip()
        self.m_status: Yield_Curve_Model_Status = Yield_Curve_Model_Status.NOT_FITTED  # type: ignore[reportAttributeAccessIssue]
        self.m_parameters: dict[str, float] = {}

        logger.info(
            "Yield_Curve_Model_Base created: '%s'",
            self.m_name_model,
        )

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> Yield_Curve_Model_Base:
        """Fit (calibrate) the model to observed yield-curve data.

        Concrete subclasses must:

        1. Validate the incoming ``data`` DataFrame.
        2. Estimate model-specific parameters.
        3. Set ``self.m_status = Yield_Curve_Model_Status.FITTED``.
        4. Return ``self`` for method chaining.

        Parameters
        ----------
        data : pl.DataFrame
            Observed yield-curve data.  If non-empty, must contain at
            minimum a ``"Maturity"`` column (positive float, years) and a
            ``"Yield"`` column (decimal-form spot yields).  Pass an empty
            ``pl.DataFrame()`` to skip estimation and use
            constructor-supplied parameters directly.

        Returns
        -------
        Yield_Curve_Model_Base
            The fitted model instance (``self``).

        Raises
        ------
        Exception_Validation_Input
            If ``data`` is missing required columns or has invalid values.
        """

    @abstractmethod
    def predict(
        self,
        maturities: list[float] | np.ndarray | None = None,
        *,
        n_points: int = 50,
        max_maturity: float = 30.0,
    ) -> pl.DataFrame:
        r"""Evaluate the fitted curve at the specified maturities.

        Concrete subclasses must:

        1. Verify the model is fitted.
        2. Evaluate the model yield at each requested maturity.
        3. Return a tidy :class:`polars.DataFrame`.

        Parameters
        ----------
        maturities : list[float] or np.ndarray or None, optional
            Specific maturities (years, all > 0) at which to evaluate the
            curve.  If ``None``, a logarithmically-spaced grid of
            ``n_points`` values in the interval
            :math:`[0.25,\, \text{max\_maturity}]` is generated.
        n_points : int, keyword-only, optional
            Number of grid points when ``maturities`` is ``None``
            (default ``50``).  Ignored if ``maturities`` is provided.
        max_maturity : float, keyword-only, optional
            Upper bound of the default maturity grid in years
            (default ``30.0``).  Ignored if ``maturities`` is provided.

        Returns
        -------
        pl.DataFrame
            DataFrame with at minimum:

            * ``"Maturity"`` -- :class:`polars.Float64`, time to maturity
              in years
            * ``"Yield"`` -- :class:`polars.Float64`, model spot yield in
              decimal form

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If ``n_points`` is not a positive integer, ``max_maturity``
            is not a positive float, or ``maturities`` contains invalid
            values.
        """

    @abstractmethod
    def get_par_yield(self, maturity: float) -> float:
        """Return the model spot yield at a single maturity.

        Parameters
        ----------
        maturity : float
            Time to maturity in years (must be > 0).

        Returns
        -------
        float
            Model spot yield in decimal form.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If ``maturity`` is not a positive finite float.
        """

    @abstractmethod
    def get_forward_rate(self, maturity: float) -> float:
        r"""Return the instantaneous forward rate at the given maturity.

        The instantaneous forward rate :math:`f(\tau)` is related to
        the spot yield :math:`y(\tau)` and zero-coupon discount factor
        :math:`P(\tau) = e^{-y(\tau)\,\tau}` by:

        .. math::

            f(\tau)
            = -\frac{\mathrm{d}}{\mathrm{d}\tau} \ln P(\tau)
            = y(\tau) + \tau \, y'(\tau)

        Parameters
        ----------
        maturity : float
            Time to maturity in years (must be > 0).

        Returns
        -------
        float
            Instantaneous forward rate in decimal form.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If ``maturity`` is not a positive finite float.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name_model(self) -> str:
        """Str : Human-readable model label."""
        return self.m_name_model

    @property
    def status(self) -> Yield_Curve_Model_Status:
        """Yield_Curve_Model_Status : Current lifecycle status."""
        return self.m_status

    @property
    def is_fitted(self) -> bool:
        """Bool : ``True`` iff the model has been fitted successfully."""
        return self.m_status == Yield_Curve_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]

    @property
    def parameters(self) -> dict[str, float]:
        """dict[str, float] : Model parameters (copy)."""
        return self.m_parameters.copy()

    # ------------------------------------------------------------------
    # Shared validation helpers
    # ------------------------------------------------------------------

    def _validate_maturity(self, maturity: float, field_name: str = "maturity") -> None:
        """Validate a single maturity value is a positive finite number.

        Parameters
        ----------
        maturity : float
            Maturity in years to validate.
        field_name : str, optional
            Name of the parameter for error messages (default
            ``"maturity"``).

        Raises
        ------
        Exception_Validation_Input
            If ``maturity`` is not a finite positive number.
        """
        if isinstance(maturity, bool) or not isinstance(maturity, (int, float)):
            raise Exception_Validation_Input(
                f"{field_name} must be a positive finite float",
                field_name=field_name,
                expected_type=float,
                actual_value=maturity,
            )
        fval = float(maturity)
        if not np.isfinite(fval) or fval <= 0.0:
            raise Exception_Validation_Input(
                f"{field_name} must be a positive finite float, got {maturity}",
                field_name=field_name,
                expected_type=float,
                actual_value=maturity,
            )

    def _validate_yield_curve_data(self, data: pl.DataFrame) -> None:
        """Validate structure and content of an observed yield-curve DataFrame.

        Parameters
        ----------
        data : pl.DataFrame
            Input DataFrame to validate.

        Raises
        ------
        Exception_Validation_Input
            If ``data`` is not a :class:`polars.DataFrame`, is empty,
            is missing ``"Maturity"`` or ``"Yield"`` columns, contains
            NaN or null values in those columns, or has any non-positive
            maturities.
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

        required_cols = {"Maturity", "Yield"}
        missing = required_cols - set(data.columns)
        if missing:
            raise Exception_Validation_Input(
                f"data is missing required columns: {sorted(missing)}",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value=data.columns,
            )

        if data["Maturity"].is_nan().any() or data["Maturity"].is_null().any():
            raise Exception_Validation_Input(
                "data 'Maturity' column must not contain NaN or null values",
                field_name="Maturity",
                expected_type=float,
                actual_value="NaN or null",
            )

        if data["Yield"].is_nan().any() or data["Yield"].is_null().any():
            raise Exception_Validation_Input(
                "data 'Yield' column must not contain NaN or null values",
                field_name="Yield",
                expected_type=float,
                actual_value="NaN or null",
            )

        if (data["Maturity"] <= 0).any():
            raise Exception_Validation_Input(
                "all 'Maturity' values must be positive (> 0)",
                field_name="Maturity",
                expected_type=float,
                actual_value="non-positive maturity",
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
                f"Model '{self.m_name_model}' has not been fitted -- call fit() first",
            )

    def _resolve_maturities(
        self,
        maturities: list[float] | np.ndarray | None,
        n_points: int,
        max_maturity: float,
    ) -> np.ndarray:
        """Convert a caller-supplied maturity spec into a validated float array.

        If *maturities* is provided, validate and return it as a 1-D
        ``float64`` array.  Otherwise generate a logarithmically-spaced
        grid from ``0.25`` to *max_maturity* with *n_points* points.

        Parameters
        ----------
        maturities : list[float] or np.ndarray or None
            Explicit maturity values, or ``None`` to use the grid.
        n_points : int
            Number of grid points (used only when *maturities* is ``None``).
        max_maturity : float
            Upper bound of the grid in years (used only when *maturities*
            is ``None``).

        Returns
        -------
        np.ndarray
            Validated 1-D array of positive finite maturities in years.

        Raises
        ------
        Exception_Validation_Input
            If *maturities* contains non-finite or non-positive values,
            or if *n_points* / *max_maturity* are invalid.
        """
        if maturities is not None:
            try:
                taus = np.asarray(maturities, dtype=float)
            except (TypeError, ValueError) as exc:
                raise Exception_Validation_Input(
                    "maturities must be convertible to a float array",
                    field_name="maturities",
                    expected_type=list,
                    actual_value=type(maturities).__name__,
                ) from exc

            if taus.ndim != 1 or len(taus) == 0:
                raise Exception_Validation_Input(
                    "maturities must be a non-empty 1-D array",
                    field_name="maturities",
                    expected_type=list,
                    actual_value=taus.shape,
                )

            if not np.all(np.isfinite(taus)):
                raise Exception_Validation_Input(
                    "all maturities must be finite (no NaN or inf)",
                    field_name="maturities",
                    expected_type=float,
                    actual_value="NaN or inf",
                )

            if not np.all(taus > 0.0):
                raise Exception_Validation_Input(
                    "all maturities must be positive (> 0)",
                    field_name="maturities",
                    expected_type=float,
                    actual_value="non-positive maturity",
                )

            return taus

        # --- generate logarithmically-spaced default grid ---
        if not isinstance(n_points, int) or isinstance(n_points, bool) or n_points < 1:
            raise Exception_Validation_Input(
                "n_points must be a positive integer (>= 1)",
                field_name="n_points",
                expected_type=int,
                actual_value=n_points,
            )

        if (
            isinstance(max_maturity, bool)
            or not isinstance(max_maturity, (int, float))
            or not np.isfinite(float(max_maturity))
            or float(max_maturity) <= 0.0
        ):
            raise Exception_Validation_Input(
                "max_maturity must be a positive finite float",
                field_name="max_maturity",
                expected_type=float,
                actual_value=max_maturity,
            )

        # Logarithmic spacing gives better resolution at short maturities
        from_log = np.log10(0.25)
        to_log = np.log10(float(max_maturity))
        return np.logspace(from_log, to_log, n_points)

    # ------------------------------------------------------------------
    # Shared utility methods
    # ------------------------------------------------------------------

    def get_summary_statistics(self, df_predict: pl.DataFrame) -> pl.DataFrame:
        """Compute basic summary statistics over a predicted yield-curve DataFrame.

        Parameters
        ----------
        df_predict : pl.DataFrame
            Output of :meth:`predict` with a ``"Yield"`` column.

        Returns
        -------
        pl.DataFrame
            Single-row DataFrame with columns ``"Mean"``, ``"Median"``,
            ``"Std"``, ``"Min"``, ``"Max"``, ``"P5"``, ``"P95"``.

        Raises
        ------
        Exception_Validation_Input
            If ``df_predict`` lacks a ``"Yield"`` column.
        """
        if "Yield" not in df_predict.columns:
            raise Exception_Validation_Input(
                "df_predict must contain a 'Yield' column",
                field_name="df_predict",
                expected_type=pl.DataFrame,
                actual_value=df_predict.columns,
            )

        yields = df_predict["Yield"].to_numpy()

        return pl.DataFrame(
            {
                "Mean": [float(np.mean(yields))],
                "Median": [float(np.median(yields))],
                "Std": [float(np.std(yields, ddof=1)) if len(yields) > 1 else 0.0],
                "Min": [float(np.min(yields))],
                "Max": [float(np.max(yields))],
                "P5": [float(np.percentile(yields, 5))],
                "P95": [float(np.percentile(yields, 95))],
            },
        )

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Yield_Curve_Model_Base(name='{self.m_name_model}', status={self.m_status.value})"
