r"""Constant (flat) yield curve model for the QWIM fixed-income pipeline.

=========================================================

Provides :class:`Yield_Curve_Model_Constant`, a deterministic model that
assumes a **flat yield curve** -- the same spot yield for every
maturity:

.. math::

    y(\tau) = r_{\mathrm{flat}} \quad \forall\; \tau > 0

Under this assumption:

* **Spot yield**: :math:`y(\tau) = r`
* **Instantaneous forward rate**: :math:`f(\tau) = r` (constant)
* **Zero-coupon price**: :math:`P(\tau) = e^{-r \tau}`

A flat curve is the simplest possible yield-curve representation and
serves as a baseline or stress-scenario anchor in fixed-income
valuation, liability discounting, and retirement-income models.
When historical ``(maturity, yield)`` data is supplied to
:meth:`fit`, the model estimates :math:`r` as the arithmetic mean of
the observed yields; otherwise it uses the constructor-supplied value
directly.

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

from src.models.yield_curve.model_yield_curve_base import (
    Yield_Curve_Model_Base,
    Yield_Curve_Model_Status,
)
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)

# ======================================================================
# Module-level defaults
# ======================================================================

_DEFAULT_FLAT_RATE: float = 0.03  # 3 % (moderate yield environment)

# Allow slight negative yields (Japan/Europe) and cap at hyperinflation level
_FLAT_RATE_MIN: float = -0.10  # lower bound: -10 %
_FLAT_RATE_MAX: float = 0.30  # upper bound:  30 %


# ======================================================================
# Constant / flat yield curve model
# ======================================================================


class Yield_Curve_Model_Constant(Yield_Curve_Model_Base):  # noqa: N801
    r"""Deterministic flat yield curve model.

    Projects the same spot yield :math:`r` for every maturity
    :math:`\tau`:

    .. math::

        y(\tau) = r \quad \forall\; \tau > 0

    Under this flat-curve assumption:

    * **Instantaneous forward rate**: :math:`f(\tau) = r`
    * **Zero-coupon price**: :math:`P(\tau) = e^{-r \tau}`

    Parameters
    ----------
    flat_rate : float, optional
        Fixed spot yield in decimal form
        (default ``0.03`` for 3 %).  Used when ``fit()`` is called
        without data or when acting as a hard-coded curve assumption.
    name_model : str, optional
        Human-readable label (default ``"Flat Yield Curve"``).

    Examples
    --------
    Hard-coded rate, no fitting required:

    >>> model = Yield_Curve_Model_Constant(flat_rate=0.04)
    >>> model.fit(pl.DataFrame())  # no-op fit
    Yield_Curve_Model_Constant(name='Flat Yield Curve', flat_rate=0.040000, status=Fitted)
    >>> df = model.predict(maturities=[1.0, 5.0, 10.0])
    >>> df["Yield"].to_list()
    [0.04, 0.04, 0.04]

    Fit to observed market yields:

    >>> import polars as pl
    >>> market_data = pl.DataFrame(
    ...     {
    ...         "Maturity": [1.0, 2.0, 5.0, 10.0],
    ...         "Yield": [0.030, 0.032, 0.036, 0.038],
    ...     }
    ... )
    >>> model = Yield_Curve_Model_Constant()
    >>> model.fit(market_data)
    Yield_Curve_Model_Constant(name='Flat Yield Curve', flat_rate=0.034000, status=Fitted)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        flat_rate: float = _DEFAULT_FLAT_RATE,
        name_model: str = "Flat Yield Curve",
    ) -> None:
        super().__init__(name_model=name_model)

        if isinstance(flat_rate, bool) or not isinstance(flat_rate, (int, float)):
            raise Exception_Validation_Input(
                "flat_rate must be a finite float",
                field_name="flat_rate",
                expected_type=float,
                actual_value=flat_rate,
            )
        if not np.isfinite(float(flat_rate)):
            raise Exception_Validation_Input(
                "flat_rate must be finite (not NaN or inf)",
                field_name="flat_rate",
                expected_type=float,
                actual_value=flat_rate,
            )
        if float(flat_rate) < _FLAT_RATE_MIN or float(flat_rate) > _FLAT_RATE_MAX:
            raise Exception_Validation_Input(
                f"flat_rate must be in [{_FLAT_RATE_MIN}, {_FLAT_RATE_MAX}]",
                field_name="flat_rate",
                expected_type=float,
                actual_value=flat_rate,
            )

        self.m_flat_rate: float = float(flat_rate)

        logger.info(
            "Yield_Curve_Model_Constant constructed with flat_rate=%.6f",
            self.m_flat_rate,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> Yield_Curve_Model_Constant:
        """Fit the flat-curve model to observed yield data.

        If *data* is a non-empty DataFrame with the required columns,
        the model updates :attr:`m_flat_rate` to the arithmetic mean of
        the observed ``"Yield"`` column.  If an empty DataFrame is passed
        the constructor value is kept unchanged.

        Parameters
        ----------
        data : pl.DataFrame
            Observed yield-curve data.  If non-empty, must contain
            columns ``"Maturity"`` (positive float, years) and
            ``"Yield"`` (decimal-form spot yields, may be negative).
            Pass an empty ``pl.DataFrame()`` to skip estimation and use
            the constructor-supplied rate.

        Returns
        -------
        Yield_Curve_Model_Constant
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
            self._validate_yield_curve_data(data)

            yield_values = data["Yield"].to_numpy()

            if not np.all(np.isfinite(yield_values)):
                raise Exception_Validation_Input(
                    "Yield column contains non-finite values",
                    field_name="Yield",
                    expected_type=float,
                    actual_value="non-finite entry",
                )

            self.m_flat_rate = float(np.mean(yield_values))
            logger.info(
                "Yield_Curve_Model_Constant fitted from %d observations; estimated flat_rate=%.6f",
                len(yield_values),
                self.m_flat_rate,
            )
        else:
            logger.info(
                "Yield_Curve_Model_Constant: empty data -- using constructor flat_rate=%.6f",
                self.m_flat_rate,
            )

        self.m_status = Yield_Curve_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]
        self.m_parameters = {"flat_rate": self.m_flat_rate}
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        maturities: list[float] | np.ndarray | None = None,
        *,
        n_points: int = 50,
        max_maturity: float = 30.0,
    ) -> pl.DataFrame:
        """Evaluate the flat yield curve at the specified maturities.

        Parameters
        ----------
        maturities : list[float] or np.ndarray or None, optional
            Explicit maturities in years (all must be > 0).  If ``None``,
            a logarithmically-spaced grid of *n_points* values up to
            *max_maturity* is generated.
        n_points : int, keyword-only, optional
            Grid size when *maturities* is ``None`` (default ``50``).
        max_maturity : float, keyword-only, optional
            Upper maturity bound in years when *maturities* is ``None``
            (default ``30.0``).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            * ``"Maturity"`` -- :class:`polars.Float64`
            * ``"Yield"``    -- :class:`polars.Float64` (all equal to
              :attr:`m_flat_rate`)

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *maturities*, *n_points*, or *max_maturity* are invalid.
        """
        self._check_is_fitted()
        taus = self._resolve_maturities(maturities, n_points, max_maturity)
        n = len(taus)

        logger.info(
            "Yield_Curve_Model_Constant.predict: n_points=%d, flat_rate=%.6f",
            n,
            self.m_flat_rate,
        )

        return pl.DataFrame(
            {
                "Maturity": taus.tolist(),
                "Yield": [self.m_flat_rate] * n,
            },
        )

    # ------------------------------------------------------------------
    # Par yield and forward rate
    # ------------------------------------------------------------------

    def get_par_yield(self, maturity: float) -> float:
        """Return the model spot yield at a single maturity.

        For the flat-curve model the result is always :attr:`m_flat_rate`,
        independent of *maturity*.

        Parameters
        ----------
        maturity : float
            Time to maturity in years (must be > 0).

        Returns
        -------
        float
            Flat yield in decimal form.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *maturity* is not a positive finite float.
        """
        self._check_is_fitted()
        self._validate_maturity(maturity)
        return self.m_flat_rate

    def get_forward_rate(self, maturity: float) -> float:
        r"""Return the instantaneous forward rate at *maturity*.

        For a flat-curve model the instantaneous forward rate equals
        the spot yield at every :math:`\tau`:

        .. math::

            f(\tau) = r \quad \forall\; \tau > 0

        Parameters
        ----------
        maturity : float
            Time to maturity in years (must be > 0).

        Returns
        -------
        float
            Flat forward rate in decimal form.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *maturity* is not a positive finite float.
        """
        self._check_is_fitted()
        self._validate_maturity(maturity)
        return self.m_flat_rate

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Yield_Curve_Model_Constant("
            f"name='{self.m_name_model}', "
            f"flat_rate={self.m_flat_rate:.6f}, "
            f"status={self.m_status.value})"
        )
