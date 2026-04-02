r"""Nelson-Siegel yield curve model for the QWIM fixed-income pipeline.

=========================================================

Provides :class:`Yield_Curve_Model_Standard`, a deterministic parametric
yield curve model based on the **Nelson-Siegel (1987)** specification --
the canonical cross-sectional term-structure model used by central banks,
government debt management offices, and the Bank for International
Settlements (BIS) for official yield curve estimation.

The Nelson-Siegel (NS) spot yield at maturity :math:`\tau` is:

.. math::

    y(\tau)
    = \beta_0
    + \beta_1 \, \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}
    + \beta_2 \left(
          \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}
          - e^{-\tau/\lambda}
      \right)

where:

* :math:`\beta_0` -- long-run yield level (:math:`y(\tau) \to \beta_0`
  as :math:`\tau \to \infty`).
* :math:`\beta_1` -- slope factor (contribution decays monotonically
  with maturity; typically negative for an upward-sloping curve).
* :math:`\beta_2` -- curvature/hump factor (captures the medium-maturity
  hump; can be positive or negative).
* :math:`\lambda > 0` -- decay factor controlling how quickly factors 2
  and 3 decay with maturity; the curvature hump peaks at
  :math:`\tau^* = \lambda`.

The corresponding **instantaneous forward rate** has the closed-form
expression:

.. math::

    f(\tau)
    = \beta_0
    + \beta_1 \, e^{-\tau/\lambda}
    + \beta_2 \, \frac{\tau}{\lambda} \, e^{-\tau/\lambda}

Fitting procedure
-----------------
For any fixed :math:`\lambda`, the NS spot-yield equation is **linear**
in :math:`(\beta_0, \beta_1, \beta_2)`, so ordinary least-squares (OLS)
yields the globally optimal betas.  Lambda is estimated by a
**grid search** over a pre-defined set of candidate values; the lambda
that minimises the sum of squared residuals is chosen.

If every lambda on the grid produces an OLS estimate outside the
expected range, the model falls back to the constructor prior values.

References
----------
* Nelson, C.R., & Siegel, A.F. (1987). Parsimonious modeling of yield
  curves. *Journal of Business*, 60(4), 473-489.
* BIS Papers No. 25 (2005). Zero-coupon yield curves: technical
  documentation.  Bank for International Settlements, Basel.
* Diebold, F.X., & Li, C. (2006). Forecasting the term structure of
  government bond yields. *Journal of Econometrics*, 130(2), 337-364.

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
# Module-level defaults and constants
# ======================================================================

# Default Nelson-Siegel parameters (approximate US Treasury normal curve)
_DEFAULT_BETA0: float = 0.05  # long-term level (5 %)
_DEFAULT_BETA1: float = -0.01  # slope: slight upward-sloping curve
_DEFAULT_BETA2: float = 0.01  # curvature: mild hump
_DEFAULT_LAMBDA: float = 2.0  # decay; hump at tau* ~ lambda ~ 2 yr

# Candidate lambda values for grid search (years).
# Grid spans practical range: hump at 0.1*lambda to 15*lambda years.
_LAMBDA_GRID: np.ndarray = np.array(
    [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
    dtype=np.float64,
)


# ======================================================================
# Module-level validation helper (exported for testing)
# ======================================================================


def _validate_positive_float(value: float, name: str) -> None:
    """Validate that *value* is a strictly positive, finite float.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Parameter name used in the error message.

    Raises
    ------
    Exception_Validation_Input
        If *value* is not a finite number strictly greater than zero.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Exception_Validation_Input(
            f"{name} must be a strictly positive finite float",
            field_name=name,
            expected_type=float,
            actual_value=value,
        )
    fval = float(value)
    if not np.isfinite(fval) or fval <= 0.0:
        raise Exception_Validation_Input(
            f"{name} must be strictly positive and finite, got {value}",
            field_name=name,
            expected_type=float,
            actual_value=value,
        )


def _validate_finite_float(value: float, name: str) -> None:
    """Validate that *value* is a finite float (may be negative or zero).

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Parameter name used in the error message.

    Raises
    ------
    Exception_Validation_Input
        If *value* is not a finite number (NaN, inf, or wrong type).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Exception_Validation_Input(
            f"{name} must be a finite float",
            field_name=name,
            expected_type=float,
            actual_value=value,
        )
    if not np.isfinite(float(value)):
        raise Exception_Validation_Input(
            f"{name} must be finite (not NaN or inf), got {value}",
            field_name=name,
            expected_type=float,
            actual_value=value,
        )


# ======================================================================
# Nelson-Siegel helper functions
# ======================================================================


def _ns_loadings(tau: float, lam: float) -> tuple[float, float, float]:
    r"""Compute the three Nelson-Siegel factor loadings at maturity *tau*.

    Parameters
    ----------
    tau : float
        Time to maturity in years (> 0).
    lam : float
        NS decay factor :math:`\lambda` (> 0).

    Returns
    -------
    tuple[float, float, float]
        ``(L1, L2, L3)`` -- the level, slope, and curvature loadings.
    """
    x = tau / lam
    # L'Hopital: lim_{x->0} (1 - e^-x)/x = 1
    l2 = 1.0 if abs(x) < 1e-10 else (1.0 - np.exp(-x)) / x
    l3 = l2 - np.exp(-x)
    return 1.0, l2, l3


def _ns_yield(tau: float, beta0: float, beta1: float, beta2: float, lam: float) -> float:
    r"""Evaluate the Nelson-Siegel spot yield at maturity *tau*.

    .. math::

        y(\tau) = \beta_0 + \beta_1 L_2(\tau) + \beta_2 L_3(\tau)

    Parameters
    ----------
    tau : float
        Maturity in years.
    beta0, beta1, beta2 : float
        Nelson-Siegel level, slope, and curvature parameters.
    lam : float
        Decay factor :math:`\lambda` (> 0).

    Returns
    -------
    float
        Spot yield in decimal form.
    """
    _, l2, l3 = _ns_loadings(tau, lam)
    return beta0 + beta1 * l2 + beta2 * l3


def _ns_forward(tau: float, beta0: float, beta1: float, beta2: float, lam: float) -> float:
    r"""Evaluate the Nelson-Siegel instantaneous forward rate at maturity *tau*.

    .. math::

        f(\tau) = \beta_0
               + \beta_1 \, e^{-\tau/\lambda}
               + \beta_2 \, \frac{\tau}{\lambda} \, e^{-\tau/\lambda}

    Parameters
    ----------
    tau : float
        Maturity in years.
    beta0, beta1, beta2 : float
        Nelson-Siegel parameters.
    lam : float
        Decay factor.

    Returns
    -------
    float
        Instantaneous forward rate in decimal form.
    """
    x = tau / lam
    exp_x = np.exp(-x)
    return beta0 + beta1 * exp_x + beta2 * x * exp_x


def _design_matrix(taus: np.ndarray, lam: float) -> np.ndarray:
    """Build the NS design matrix for a vector of maturities.

    Parameters
    ----------
    taus : np.ndarray
        1-D array of maturities in years.
    lam : float
        Candidate decay factor.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(taus), 3)`` with columns
        ``[1, L2(tau), L3(tau)]``.
    """
    rows = [_ns_loadings(float(t), lam) for t in taus]
    return np.array(rows, dtype=np.float64)


# ======================================================================
# Nelson-Siegel yield curve model
# ======================================================================


class Yield_Curve_Model_Standard(Yield_Curve_Model_Base):  # noqa: N801
    r"""Nelson-Siegel parametric yield curve model.

    Implements the three-factor term-structure specification:

    .. math::

        y(\tau)
        = \beta_0
        + \beta_1
          \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}
        + \beta_2
          \left(
              \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}
              - e^{-\tau/\lambda}
          \right)

    Parameters
    ----------
    beta0 : float, optional
        Initial guess for the long-term level parameter
        (default ``0.05``).
    beta1 : float, optional
        Initial guess for the slope parameter (default ``-0.01``).
    beta2 : float, optional
        Initial guess for the curvature parameter (default ``0.01``).
    lambda_ : float, optional
        Initial guess for the decay factor :math:`\lambda > 0`
        (default ``2.0``, placing the hump at approx. 2 years).
        Overwritten by :meth:`fit`.
    name_model : str, optional
        Human-readable label (default ``"Nelson-Siegel"``).

    Attributes
    ----------
    m_beta0, m_beta1, m_beta2 : float
        Fitted Nelson-Siegel level, slope, and curvature parameters.
    m_lambda : float
        Fitted decay factor.

    Notes
    -----
    * :math:`\lambda` must be strictly positive; the beta parameters
      can take any finite real value.
    * The fitter requires at least **3** data points.

    Examples
    --------
    >>> import polars as pl
    >>> market = pl.DataFrame(
    ...     {
    ...         "Maturity": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
    ...         "Yield": [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.048, 0.050],
    ...     }
    ... )
    >>> model = Yield_Curve_Model_Standard()
    >>> model.fit(market)  # doctest: +SKIP
    >>> df = model.predict(maturities=[1.0, 5.0, 10.0, 30.0])
    >>> "Yield" in df.columns
    True
    >>> "forward_rate" in df.columns
    True
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        beta0: float = _DEFAULT_BETA0,
        beta1: float = _DEFAULT_BETA1,
        beta2: float = _DEFAULT_BETA2,
        lambda_: float = _DEFAULT_LAMBDA,
        name_model: str = "Nelson-Siegel",
    ) -> None:
        super().__init__(name_model=name_model)

        # lambda must be strictly positive (appears in denominator)
        _validate_positive_float(lambda_, "lambda_")

        # betas can be any finite real number
        _validate_finite_float(beta0, "beta0")
        _validate_finite_float(beta1, "beta1")
        _validate_finite_float(beta2, "beta2")

        self.m_beta0: float = float(beta0)
        self.m_beta1: float = float(beta1)
        self.m_beta2: float = float(beta2)
        self.m_lambda: float = float(lambda_)

        logger.info(
            "Yield_Curve_Model_Standard constructed: beta0=%.4f, beta1=%.4f, "
            "beta2=%.4f, lambda=%.4f",
            self.m_beta0,
            self.m_beta1,
            self.m_beta2,
            self.m_lambda,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> Yield_Curve_Model_Standard:
        r"""Estimate NS parameters from observed ``(maturity, yield)`` pairs.

        For each candidate :math:`\lambda` in the pre-defined grid
        :data:`_LAMBDA_GRID`, the factor loadings are computed, linearised
        OLS is applied to estimate :math:`(\beta_0, \beta_1, \beta_2)`,
        and the residual sum of squares is recorded.  The lambda that
        produces the lowest SSR is selected, and the corresponding OLS
        estimates become the fitted parameters.

        If every lambda yields a non-positive beta0 (long-term yield), the
        lambda with the smallest SSR is still accepted and beta0 is used
        as fitted (the model does not impose a positivity constraint on
        beta0 after fitting, since negative long-term yields are
        financially plausible in some environments).  If OLS fails
        completely (singular design matrix), the model falls back to
        the constructor prior values.

        Parameters
        ----------
        data : pl.DataFrame
            Observed yield-curve data with columns ``"Maturity"``
            (positive float, years) and ``"Yield"`` (decimal-form spot
            yields; may be negative).  Must contain at least 3 rows.
            If an empty ``pl.DataFrame()`` is passed, the constructor
            prior values are used directly.

        Returns
        -------
        Yield_Curve_Model_Standard
            The fitted model instance (``self``).

        Raises
        ------
        Exception_Validation_Input
            If *data* is invalid, contains non-finite values, or has
            fewer than 3 rows.
        """
        if not isinstance(data, pl.DataFrame):
            raise Exception_Validation_Input(
                "data must be a polars.DataFrame",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value=type(data).__name__,
            )

        if len(data) == 0:
            # Empty DataFrame -- use prior parameters
            logger.info(
                "Yield_Curve_Model_Standard: empty data -- using constructor priors.",
            )
            self.m_status = Yield_Curve_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]
            self.m_parameters = {
                "beta0": self.m_beta0,
                "beta1": self.m_beta1,
                "beta2": self.m_beta2,
                "lambda": self.m_lambda,
            }
            return self

        self._validate_yield_curve_data(data)

        # Sort by maturity, convert to float64
        data = data.sort("Maturity")
        taus = data["Maturity"].to_numpy().astype(np.float64)
        yields = data["Yield"].to_numpy().astype(np.float64)

        if not np.all(np.isfinite(yields)):
            raise Exception_Validation_Input(
                "Yield column contains non-finite values",
                field_name="Yield",
                expected_type=float,
                actual_value="non-finite entry",
            )

        n = len(taus)
        if n < 3:
            raise Exception_Validation_Input(
                "data must have at least 3 rows to fit Nelson-Siegel parameters",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value=n,
            )

        # --- grid search over lambda ---
        best_ssr = np.inf
        best_betas: np.ndarray | None = None
        best_lambda: float = self.m_lambda

        for lam in _LAMBDA_GRID:
            xmat = _design_matrix(taus, float(lam))
            try:
                betas, _, rank, _ = lstsq(xmat, yields, rcond=None)
            except np.linalg.LinAlgError:
                continue

            if rank < 3:
                # Rank-deficient; skip this lambda
                continue

            # Compute SSR explicitly (lstsq residuals may be empty for
            # underdetermined systems)
            y_hat = xmat @ betas
            ssr = float(np.sum((yields - y_hat) ** 2))

            if ssr < best_ssr:
                best_ssr = ssr
                best_betas = betas.copy()
                best_lambda = float(lam)

        if best_betas is None:
            # All lambda candidates failed -- fall back to priors
            logger.warning(
                "Yield_Curve_Model_Standard: all lambda candidates produced "
                "rank-deficient OLS; falling back to constructor priors.",
            )
        else:
            if not np.all(np.isfinite(best_betas)):
                logger.warning(
                    "Yield_Curve_Model_Standard: OLS produced non-finite betas; "
                    "falling back to constructor priors.",
                )
            else:
                self.m_beta0 = float(best_betas[0])
                self.m_beta1 = float(best_betas[1])
                self.m_beta2 = float(best_betas[2])
                self.m_lambda = best_lambda

        self.m_status = Yield_Curve_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]
        self.m_parameters = {
            "beta0": self.m_beta0,
            "beta1": self.m_beta1,
            "beta2": self.m_beta2,
            "lambda": self.m_lambda,
        }

        logger.info(
            "Yield_Curve_Model_Standard fitted: beta0=%.4f, beta1=%.4f, "
            "beta2=%.4f, lambda=%.4f (SSR=%.6e, n=%d)",
            self.m_beta0,
            self.m_beta1,
            self.m_beta2,
            self.m_lambda,
            best_ssr,
            n,
        )

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
        r"""Evaluate the Nelson-Siegel curve at the specified maturities.

        Returns the spot yield and instantaneous forward rate (per the
        analytical closed-form expressions) at each requested maturity.

        Parameters
        ----------
        maturities : list[float] or np.ndarray or None, optional
            Explicit maturities in years (all > 0).  If ``None``, a
            logarithmically-spaced grid of *n_points* values up to
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

            * ``"Maturity"``    -- :class:`polars.Float64`
            * ``"Yield"``       -- :class:`polars.Float64`, NS spot yield
            * ``"forward_rate"``-- :class:`polars.Float64`, instantaneous
              forward rate

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *maturities*, *n_points*, or *max_maturity* are invalid.
        """
        self._check_is_fitted()
        taus = self._resolve_maturities(maturities, n_points, max_maturity)

        beta0, beta1, beta2, lam = (
            self.m_beta0,
            self.m_beta1,
            self.m_beta2,
            self.m_lambda,
        )

        yield_col: list[float] = [_ns_yield(float(t), beta0, beta1, beta2, lam) for t in taus]
        fwd_col: list[float] = [_ns_forward(float(t), beta0, beta1, beta2, lam) for t in taus]

        logger.info(
            "Yield_Curve_Model_Standard.predict: n_points=%d",
            len(taus),
        )

        return pl.DataFrame(
            {
                "Maturity": taus.tolist(),
                "Yield": yield_col,
                "forward_rate": fwd_col,
            },
        )

    # ------------------------------------------------------------------
    # Par yield and forward rate
    # ------------------------------------------------------------------

    def get_par_yield(self, maturity: float) -> float:
        """Return the Nelson-Siegel spot yield at a single maturity.

        Parameters
        ----------
        maturity : float
            Time to maturity in years (must be > 0).

        Returns
        -------
        float
            NS spot yield in decimal form.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *maturity* is not a positive finite float.
        """
        self._check_is_fitted()
        self._validate_maturity(maturity)
        return _ns_yield(
            float(maturity),
            self.m_beta0,
            self.m_beta1,
            self.m_beta2,
            self.m_lambda,
        )

    def get_forward_rate(self, maturity: float) -> float:
        r"""Return the Nelson-Siegel instantaneous forward rate at *maturity*.

        Uses the closed-form expression:

        .. math::

            f(\tau)
            = \beta_0
            + \beta_1 \, e^{-\tau/\lambda}
            + \beta_2 \, \frac{\tau}{\lambda} \, e^{-\tau/\lambda}

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
            If *maturity* is not a positive finite float.
        """
        self._check_is_fitted()
        self._validate_maturity(maturity)
        return _ns_forward(
            float(maturity),
            self.m_beta0,
            self.m_beta1,
            self.m_beta2,
            self.m_lambda,
        )

    # ------------------------------------------------------------------
    # Curve shape diagnostics
    # ------------------------------------------------------------------

    def get_slope(self) -> float:
        r"""Return the Nelson-Siegel slope parameter :math:`\beta_1`.

        The slope captures the difference between the long-run level and
        the instantaneous short rate:

        .. math::

            f(0^+) - \beta_0 = \beta_1

        A negative :math:`\beta_1` corresponds to an upward-sloping curve.

        Returns
        -------
        float
            Slope parameter :math:`\beta_1`.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return self.m_beta1

    def get_curvature(self) -> float:
        r"""Return the Nelson-Siegel curvature parameter :math:`\beta_2`.

        A positive :math:`\beta_2` produces a hump (concave-up curve);
        a negative value produces an inverted hump (concave-down).

        Returns
        -------
        float
            Curvature parameter :math:`\beta_2`.

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return self.m_beta2

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Yield_Curve_Model_Standard("
            f"name='{self.m_name_model}', "
            f"beta0={self.m_beta0:.4f}, "
            f"beta1={self.m_beta1:.4f}, "
            f"beta2={self.m_beta2:.4f}, "
            f"lambda={self.m_lambda:.4f}, "
            f"status={self.m_status.value})"
        )
