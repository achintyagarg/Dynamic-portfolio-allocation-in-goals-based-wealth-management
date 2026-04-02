r"""Standard Monte Carlo portfolio simulation.

========================================================

Implements :class:`Simulation_Standard`, a concrete subclass of
:class:`Simulation_Base` that runs a standard Monte Carlo simulation
over a QWIM portfolio.

The simulation pipeline:

1. Generate (or receive) scenario returns via
   :class:`Scenarios_Distribution`.
2. For each scenario path compute the weighted portfolio return
   at each step.
3. Compound those returns into portfolio value paths starting from
   an initial investment.
4. Store everything in a :class:`polars.DataFrame`.

Formulae
--------

Portfolio return at each date $t$ for scenario $s$:

$$
r_{p,t}^{(s)} = \sum_{k=1}^{K} w_k \, r_{k,t}^{(s)}
$$

Portfolio value path:

$$
V_t^{(s)} = V_0 \prod_{j=1}^{t}\bigl(1 + r_{p,j}^{(s)}\bigr)
$$

Author
------
QWIM Team

Version
-------
0.6.0 (2026-03-01)
"""

from __future__ import annotations

import datetime as dt

from typing import TYPE_CHECKING

import numpy as np
import polars as pl


if TYPE_CHECKING:
    from collections.abc import Sequence

from src.models.simulation.model_simulation_base import (
    Simulation_Base,
    Simulation_Status,
)
from src.num_methods.scenarios.scenarios_distrib import (
    Distribution_Type,
    Scenarios_Distribution,
)
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)


# ======================================================================
# Default constants
# ======================================================================

DEFAULT_NUM_SCENARIOS: int = 1_000
"""Default number of Monte Carlo scenario paths."""

DEFAULT_NUM_DAYS: int = 252
"""Default simulation horizon (one trading year)."""

DEFAULT_INITIAL_VALUE: float = 100.0
"""Default starting portfolio value."""

DEFAULT_RANDOM_SEED: int = 42
"""Default RNG seed for reproducibility."""


# ======================================================================
# Simulation_Standard
# ======================================================================


class Simulation_Standard(Simulation_Base):  # noqa: N801
    r"""Standard Monte Carlo simulation for a QWIM portfolio.

    Given scenario returns from :class:`Scenarios_Distribution` and
    static portfolio weights, this class compounds per-period returns
    across all scenario paths to produce a fan of portfolio value
    trajectories.

    Parameters
    ----------
    names_components : Sequence[str]
        Component (asset) ticker symbols.
    weights : np.ndarray
        Static portfolio weights of shape ``(K,)`` summing to 1.
    distribution_type : Distribution_Type
        Distribution to sample from.
    mean_returns : np.ndarray | None
        Mean daily returns per component.  ``None`` → zeros.
    covariance_matrix : np.ndarray | None
        ``(K, K)`` covariance matrix.  ``None`` → identity.
    initial_value : float
        Starting portfolio value.
    num_scenarios : int
        Number of Monte Carlo paths.
    num_days : int
        Simulation horizon in trading days.
    start_date : dt.date | None
        First simulation date.  ``None`` → today.
    random_seed : int | None
        RNG seed.
    degrees_of_freedom : float
        DoF for Student-*t* (ignored for Normal / Lognormal).
    name_simulation : str
        Human-readable label.
    """

    def __init__(
        self,
        names_components: Sequence[str],
        weights: np.ndarray,
        distribution_type: Distribution_Type = Distribution_Type.NORMAL,  # type: ignore[reportArgumentType]
        mean_returns: np.ndarray | None = None,
        covariance_matrix: np.ndarray | None = None,
        initial_value: float = DEFAULT_INITIAL_VALUE,
        num_scenarios: int = DEFAULT_NUM_SCENARIOS,
        num_days: int = DEFAULT_NUM_DAYS,
        start_date: dt.date | None = None,
        random_seed: int | None = DEFAULT_RANDOM_SEED,
        degrees_of_freedom: float = 5.0,
        name_simulation: str = "Standard Monte Carlo",
    ) -> None:
        # --- validate components ---
        if names_components is None or len(names_components) == 0:
            raise Exception_Validation_Input(
                "names_components must be non-empty",
                field_name="names_components",
                expected_type=list,
                actual_value=names_components,
            )

        num_k = len(names_components)

        # --- validate distribution type ---
        if not isinstance(distribution_type, Distribution_Type):
            raise Exception_Validation_Input(
                "distribution_type must be Distribution_Type enum",
                field_name="distribution_type",
                expected_type=Distribution_Type,
                actual_value=type(distribution_type).__name__,
            )

        # --- default mean / covariance ---
        if mean_returns is None:
            mean_returns = np.zeros(num_k, dtype=np.float64)

        if covariance_matrix is None:
            covariance_matrix = np.eye(num_k, dtype=np.float64) * 0.0004

        # -- store scenario parameters ---
        self.m_distribution_type: Distribution_Type = distribution_type
        self.m_mean_returns: np.ndarray = np.asarray(mean_returns, dtype=np.float64)
        self.m_covariance_matrix: np.ndarray = np.asarray(covariance_matrix, dtype=np.float64)
        self.m_num_days: int = int(num_days)
        self.m_start_date: dt.date = start_date or dt.datetime.now(tz=dt.UTC).date()
        self.m_random_seed: int | None = random_seed
        self.m_degrees_of_freedom: float = float(degrees_of_freedom)

        # Build the scenario generator (deferred — scenarios generated in run())
        self.m_scenario_generator: Scenarios_Distribution | None = None

        # Create a placeholder Scenarios_Distribution for parent validation
        placeholder = Scenarios_Distribution(
            names_components=list(names_components),
            distribution_type=distribution_type,
            mean_returns=self.m_mean_returns,
            covariance_matrix=self.m_covariance_matrix,
            degrees_of_freedom=degrees_of_freedom,
            start_date=self.m_start_date,
            num_days=num_days,
            num_scenarios=1,
            random_seed=random_seed,
        )

        super().__init__(
            scenarios=placeholder,
            names_components=list(names_components),
            weights=np.asarray(weights, dtype=np.float64),
            initial_value=initial_value,
            num_scenarios=num_scenarios,
            name_simulation=name_simulation,
        )

        logger.info(
            "Simulation_Standard created",
            extra={
                "event_type": "simulation_standard_created",
                "simulation_name": self.m_name_simulation,
                "distribution": self.m_distribution_type.value,
                "num_components": self.m_num_components,
                "num_scenarios": self.m_num_scenarios,
                "num_days": self.m_num_days,
                "initial_value": self.m_initial_value,
                "random_seed": self.m_random_seed,
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def distribution_type(self) -> Distribution_Type:
        """Distribution_Type : Distribution used for scenario generation."""
        return self.m_distribution_type

    @property
    def num_days(self) -> int:
        """Int : Simulation horizon in trading days."""
        return self.m_num_days

    @property
    def start_date(self) -> dt.date:
        """dt.date : First date of the simulation."""
        return self.m_start_date

    @property
    def random_seed(self) -> int | None:
        """Int | None : RNG seed for reproducibility."""
        return self.m_random_seed

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pl.DataFrame:
        r"""Execute the standard Monte Carlo simulation.

        Pipeline:

        1. For each of *N* scenario paths, create a
           :class:`Scenarios_Distribution` with a unique RNG seed
           derived from the base seed.
        2. Generate daily component returns of shape ``(T, K)``.
        3. Compute the weighted portfolio return:
           $r_{p,t} = \mathbf{w}^\top \mathbf{r}_t$.
        4. Compound into portfolio values:
           $V_t = V_0 \prod_{j=1}^{t}(1 + r_{p,j})$.
        5. Assemble the ``Date x N`` results DataFrame.

        Returns
        -------
        pl.DataFrame
            Columns: ``Date``, ``Scenario_1``, ``Scenario_2``, …
        """
        self.m_status = Simulation_Status.RUNNING  # type: ignore[reportAttributeAccessIssue]
        logger.info(
            "SIMULATION_START: %s -- %d scenarios x %d days",
            self.m_name_simulation,
            self.m_num_scenarios,
            self.m_num_days,
        )

        try:
            portfolio_paths = self._run_batch()
        except Exception as exc:
            self.m_status = Simulation_Status.FAILED  # type: ignore[reportAttributeAccessIssue]
            logger.exception(
                "SIMULATION_FAILED: %s -- %s",
                self.m_name_simulation,
                exc,
            )
            raise Exception_Calculation(
                f"Simulation failed: {exc}",
            ) from exc

        # --- assemble results DataFrame ---
        dates = self._generate_dates()

        data: dict[str, list[float] | list[dt.date]] = {"Date": dates}
        for idx_s in range(self.m_num_scenarios):
            data[f"Scenario_{idx_s + 1}"] = portfolio_paths[:, idx_s].tolist()

        self.m_df_results = pl.DataFrame(data).with_columns(
            pl.col("Date").cast(pl.Date),
        )

        self.m_status = Simulation_Status.COMPLETED  # type: ignore[reportAttributeAccessIssue]
        logger.info(
            "SIMULATION_COMPLETE: %s -- shape %s",
            self.m_name_simulation,
            self.m_df_results.shape,
        )

        return self.m_df_results.clone()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_dates(self) -> list[dt.date]:
        """Generate business dates for the simulation horizon.

        Returns
        -------
        list[dt.date]
            *T* business days starting from ``m_start_date``.
        """
        result: list[dt.date] = []
        current = self.m_start_date
        while len(result) < self.m_num_days:
            if current.weekday() < 5:
                result.append(current)
            current += dt.timedelta(days=1)
        return result

    def _run_batch(self) -> np.ndarray:
        r"""Run all scenario paths in a single vectorised batch.

        Instead of generating one path at a time, we generate a
        3-D array of returns ``(T, K, N)`` and compute portfolio
        returns via a single matrix-vector product.

        Returns
        -------
        np.ndarray
            Portfolio value matrix of shape ``(T, N)``.
        """
        num_t = self.m_num_days
        num_k = self.m_num_components
        num_n = self.m_num_scenarios

        base_seed = self.m_random_seed if self.m_random_seed is not None else 0
        rng = np.random.default_rng(base_seed)

        # --- generate correlated returns: (T, K, N) ---
        mat_chol = self._safe_cholesky(self.m_covariance_matrix)

        if self.m_distribution_type == Distribution_Type.NORMAL:
            raw = rng.standard_normal((num_t, num_n, num_k))
            # Apply correlation: (T, N, K) @ (K, K)^T → (T, N, K)
            correlated = raw @ mat_chol.T + self.m_mean_returns
            returns_3d = correlated  # (T, N, K)

        elif self.m_distribution_type == Distribution_Type.LOGNORMAL:
            mu = self.m_mean_returns
            cov = self.m_covariance_matrix
            # Map to underlying normal parameters
            sigma_ln = np.zeros((num_k, num_k), dtype=np.float64)
            for idx_i in range(num_k):
                for idx_j in range(num_k):
                    ratio = cov[idx_i, idx_j] / (mu[idx_i] * mu[idx_j])
                    arg = 1.0 + ratio
                    if arg <= 0:
                        raise Exception_Calculation(
                            f"Lognormal mapping failed for ({idx_i}, {idx_j})",
                        )
                    sigma_ln[idx_i, idx_j] = np.log(arg)

            mu_ln = np.log(mu) - 0.5 * np.diag(sigma_ln)
            mat_chol_ln = self._safe_cholesky(sigma_ln)

            raw = rng.standard_normal((num_t, num_n, num_k))
            normal_samples = raw @ mat_chol_ln.T + mu_ln
            # For lognormal: returns = exp(Y) - 1 (arithmetic returns)
            returns_3d = np.exp(normal_samples) - 1.0

        elif self.m_distribution_type == Distribution_Type.STUDENT_T:
            nu = self.m_degrees_of_freedom
            scale = (nu - 2.0) / nu * self.m_covariance_matrix
            mat_chol_t = self._safe_cholesky(scale)

            raw = rng.standard_normal((num_t, num_n, num_k))
            chi2 = rng.chisquare(df=nu, size=(num_t, num_n))
            scaling = np.sqrt(chi2 / nu)[..., np.newaxis]  # (T, N, 1)
            returns_3d = self.m_mean_returns + (raw @ mat_chol_t.T) / scaling

        else:
            raise Exception_Calculation(
                f"Unknown distribution: {self.m_distribution_type}",
            )

        # --- portfolio returns: (T, N) ---
        # returns_3d shape: (T, N, K), weights shape: (K,)
        portfolio_returns = returns_3d @ self.m_weights  # (T, N)

        # --- compound into values: (T, N) ---
        growth_factors = 1.0 + portfolio_returns  # (T, N)
        cumulative_growth = np.cumprod(growth_factors, axis=0)  # (T, N)
        return self.m_initial_value * cumulative_growth  # (T, N)

    @staticmethod
    def _safe_cholesky(matrix: np.ndarray) -> np.ndarray:
        """Cholesky decomposition with PSD fallback.

        Parameters
        ----------
        matrix : np.ndarray
            Symmetric matrix.

        Returns
        -------
        np.ndarray
            Lower-triangular factor L.
        """
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigvals = np.maximum(eigvals, 1e-10)
            psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            return np.linalg.cholesky(psd)

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_historical_data(
        cls,
        price_data: pl.DataFrame,
        weights: np.ndarray,
        distribution_type: Distribution_Type = Distribution_Type.NORMAL,  # type: ignore[reportArgumentType]
        initial_value: float = DEFAULT_INITIAL_VALUE,
        num_scenarios: int = DEFAULT_NUM_SCENARIOS,
        num_days: int = DEFAULT_NUM_DAYS,
        start_date: dt.date | None = None,
        random_seed: int | None = DEFAULT_RANDOM_SEED,
        degrees_of_freedom: float = 5.0,
        name_simulation: str = "Historical MC",
    ) -> Simulation_Standard:
        """Create a simulation calibrated from historical price data.

        Estimates mean returns and covariance from historical prices
        and forwards them to the standard constructor.

        Parameters
        ----------
        price_data : pl.DataFrame
            DataFrame with ``Date`` and component price columns.
        weights : np.ndarray
            Portfolio weights.
        distribution_type : Distribution_Type
            Distribution to sample from.
        initial_value : float
            Starting portfolio value.
        num_scenarios : int
            Number of paths.
        num_days : int
            Simulation horizon.
        start_date : dt.date | None
            First date (default: today).
        random_seed : int | None
            RNG seed.
        degrees_of_freedom : float
            DoF for Student-*t*.
        name_simulation : str
            Label.

        Returns
        -------
        Simulation_Standard
        """
        if "Date" not in price_data.columns:
            raise Exception_Validation_Input(
                "price_data must have a 'Date' column",
                field_name="price_data",
                expected_type=pl.DataFrame,
                actual_value=price_data.columns,
            )

        component_cols = [c for c in price_data.columns if c != "Date"]
        if len(component_cols) == 0:
            raise Exception_Validation_Input(
                "No component columns in price_data (need >= 1)",
                field_name="price_data",
                expected_type=pl.DataFrame,
                actual_value=0,
            )

        # Compute daily returns
        mat = price_data.select(component_cols).to_numpy()
        returns = np.diff(mat, axis=0) / mat[:-1]

        mean_ret = np.mean(returns, axis=0)
        cov_mat = np.cov(returns, rowvar=False)

        # For lognormal, shift means to be positive (1 + r)
        if distribution_type == Distribution_Type.LOGNORMAL:
            mean_ret = 1.0 + mean_ret

        return cls(
            names_components=component_cols,
            weights=weights,
            distribution_type=distribution_type,
            mean_returns=mean_ret,
            covariance_matrix=cov_mat,
            initial_value=initial_value,
            num_scenarios=num_scenarios,
            num_days=num_days,
            start_date=start_date,
            random_seed=random_seed,
            degrees_of_freedom=degrees_of_freedom,
            name_simulation=name_simulation,
        )

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Simulation_Standard("
            f"name='{self.m_name_simulation}', "
            f"status={self.m_status.value}, "
            f"distribution={self.m_distribution_type.value}, "
            f"components={self.m_num_components}, "
            f"scenarios={self.m_num_scenarios}, "
            f"days={self.m_num_days})"
        )
