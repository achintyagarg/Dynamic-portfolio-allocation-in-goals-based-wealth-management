r"""Base class for QWIM portfolio simulations.

========================================================

Provides :class:`Simulation_Base`, an abstract base class that runs
Monte Carlo simulations over a QWIM portfolio using scenarios generated
by a :class:`Scenarios_Base` subclass.

The simulation produces a :class:`polars.DataFrame` of simulated
portfolio values across multiple scenario paths.

Design goals
------------
* **Performance** — heavy-lifting delegated to NumPy / Polars.
* **Simplicity** — thin, Pythonic wrapper with clear validation.
* **Extensibility** — concrete subclasses override :meth:`run`.

Author
------
QWIM Team

Version
-------
0.6.0 (2026-03-01)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from aenum import Enum


if TYPE_CHECKING:
    from collections.abc import Sequence

from src.num_methods.scenarios.scenarios_base import (
    Scenarios_Base,
)
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)


# ======================================================================
# Enums
# ======================================================================


class Simulation_Status(Enum):  # noqa: N801  # type: ignore[reportGeneralTypeIssues]
    """Status of a simulation run.

    Attributes
    ----------
    NOT_STARTED : str
        Simulation has not been executed yet.
    RUNNING : str
        Simulation is currently in progress.
    COMPLETED : str
        Simulation finished successfully.
    FAILED : str
        Simulation encountered an error.
    """

    NOT_STARTED = "Not Started"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"


class Aggregation_Method(Enum):  # noqa: N801  # type: ignore[reportGeneralTypeIssues]
    """Method for aggregating scenario results.

    Attributes
    ----------
    MEAN : str
        Arithmetic mean across scenarios.
    MEDIAN : str
        Median across scenarios.
    PERCENTILE : str
        Specific percentile (e.g. 5th, 95th).
    """

    MEAN = "Mean"
    MEDIAN = "Median"
    PERCENTILE = "Percentile"


# ======================================================================
# Base class
# ======================================================================


class Simulation_Base(ABC):  # noqa: N801
    r"""Abstract base class for QWIM portfolio Monte Carlo simulations.

    A simulation takes:

    1. A set of **scenarios** (daily returns for each component).
    2. **Portfolio weights** (static or time-varying).
    3. An **initial portfolio value**.

    And produces a :class:`polars.DataFrame` of portfolio values per
    scenario path and per date.

    Parameters
    ----------
    scenarios : Scenarios_Base
        Scenario generator (must have ``generate()`` called first, or
        the simulation will call it).
    names_components : Sequence[str]
        Component (asset) names — must match the scenario columns.
    weights : np.ndarray
        Portfolio weights, shape ``(K,)`` summing to 1.0.
    initial_value : float
        Starting portfolio value (default 100.0).
    num_scenarios : int
        Number of Monte Carlo paths to simulate.
    name_simulation : str
        Human-readable label.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        scenarios: Scenarios_Base,
        names_components: Sequence[str],
        weights: np.ndarray,
        initial_value: float = 100.0,
        num_scenarios: int = 1_000,
        name_simulation: str = "QWIM Simulation",
    ) -> None:
        # --- validate scenarios ---
        if not isinstance(scenarios, Scenarios_Base):
            raise Exception_Validation_Input(
                "scenarios must be a Scenarios_Base instance",
                field_name="scenarios",
                expected_type=Scenarios_Base,
                actual_value=type(scenarios).__name__,
            )

        # --- validate components ---
        if names_components is None or len(names_components) == 0:
            raise Exception_Validation_Input(
                "names_components must be a non-empty sequence",
                field_name="names_components",
                expected_type=list,
                actual_value=names_components,
            )

        num_k = len(names_components)

        # --- validate weights ---
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape != (num_k,):
            raise Exception_Validation_Input(
                f"weights shape must be ({num_k},), got {weights_arr.shape}",
                field_name="weights",
                expected_type=np.ndarray,
                actual_value=weights_arr.shape,
            )

        weight_sum = float(np.sum(weights_arr))
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            logger.warning(
                "Weights sum to %.6f, normalizing to 1.0",
                weight_sum,
            )
            weights_arr = weights_arr / weight_sum

        # --- validate initial value ---
        if initial_value <= 0:
            raise Exception_Validation_Input(
                "initial_value must be positive (> 0)",
                field_name="initial_value",
                expected_type=float,
                actual_value=initial_value,
            )

        # --- validate num_scenarios ---
        if not isinstance(num_scenarios, int) or num_scenarios < 1:
            raise Exception_Validation_Input(
                "num_scenarios must be a positive integer (> 0)",
                field_name="num_scenarios",
                expected_type=int,
                actual_value=num_scenarios,
            )

        # --- store members ---
        self.m_scenarios: Scenarios_Base = scenarios
        self.m_names_components: list[str] = list(names_components)
        self.m_num_components: int = num_k
        self.m_weights: np.ndarray = weights_arr
        self.m_initial_value: float = float(initial_value)
        self.m_num_scenarios: int = num_scenarios
        self.m_name_simulation: str = str(name_simulation).strip()
        self.m_status: Simulation_Status = Simulation_Status.NOT_STARTED  # type: ignore[reportAttributeAccessIssue]
        self.m_df_results: pl.DataFrame | None = None

        logger.info(
            "Simulation_Base created: '%s', %d components, %d scenarios, initial=%.2f",
            self.m_name_simulation,
            self.m_num_components,
            self.m_num_scenarios,
            self.m_initial_value,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def names_components(self) -> list[str]:
        """list[str] : Component names (copy)."""
        return self.m_names_components.copy()

    @property
    def num_components(self) -> int:
        """Int : Number of portfolio components."""
        return self.m_num_components

    @property
    def weights(self) -> np.ndarray:
        """np.ndarray : Portfolio weights (copy)."""
        return self.m_weights.copy()

    @property
    def initial_value(self) -> float:
        """Float : Initial portfolio value."""
        return self.m_initial_value

    @property
    def num_scenarios(self) -> int:
        """Int : Number of Monte Carlo paths."""
        return self.m_num_scenarios

    @property
    def name_simulation(self) -> str:
        """Str : Human-readable simulation label."""
        return self.m_name_simulation

    @property
    def status(self) -> Simulation_Status:
        """Simulation_Status : Current run status."""
        return self.m_status

    @property
    def df_results(self) -> pl.DataFrame | None:
        """pl.DataFrame | None : Simulation results (None until run)."""
        if self.m_df_results is not None:
            return self.m_df_results.clone()
        return None

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> pl.DataFrame:
        """Execute the Monte Carlo simulation.

        Concrete subclasses must:

        1. Generate scenarios if not already done.
        2. Apply portfolio weights to scenario returns.
        3. Compute portfolio values across all paths.
        4. Store results in ``self.m_df_results``.

        Returns
        -------
        pl.DataFrame
            Simulation results with ``Date`` column and scenario
            value columns.
        """

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def get_summary_statistics(self) -> pl.DataFrame:
        """Compute summary statistics across all simulation paths.

        Returns a DataFrame with ``Date``, ``Mean``, ``Median``,
        ``Std``, ``P5``, ``P25``, ``P75``, ``P95``, ``Min``, ``Max``.

        Returns
        -------
        pl.DataFrame
        """
        if self.m_df_results is None:
            raise Exception_Calculation(
                "Simulation has not been run yet — call run() first",
            )

        df = self.m_df_results
        scenario_cols = [c for c in df.columns if c != "Date"]

        if len(scenario_cols) == 0:
            raise Exception_Calculation("No scenario columns found in results")

        mat = df.select(scenario_cols).to_numpy()

        stats_data: dict[str, list[float]] = {
            "Mean": np.mean(mat, axis=1).tolist(),
            "Median": np.median(mat, axis=1).tolist(),
            "Std": np.std(mat, axis=1, ddof=1).tolist(),
            "P5": np.percentile(mat, 5, axis=1).tolist(),
            "P25": np.percentile(mat, 25, axis=1).tolist(),
            "P75": np.percentile(mat, 75, axis=1).tolist(),
            "P95": np.percentile(mat, 95, axis=1).tolist(),
            "Min": np.min(mat, axis=1).tolist(),
            "Max": np.max(mat, axis=1).tolist(),
        }

        return pl.DataFrame({"Date": df["Date"]}).with_columns(
            [pl.Series(name, values) for name, values in stats_data.items()],
        )

    def get_terminal_values(self) -> pl.DataFrame:
        """Extract the final portfolio value from each scenario path.

        Returns
        -------
        pl.DataFrame
            Single-row DataFrame of terminal values per scenario.
        """
        if self.m_df_results is None:
            raise Exception_Calculation(
                "Simulation has not been run yet — call run() first",
            )

        return self.m_df_results.tail(1).select(
            [c for c in self.m_df_results.columns if c != "Date"],
        )

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Simulation_Base("
            f"name='{self.m_name_simulation}', "
            f"status={self.m_status.value}, "
            f"components={self.m_num_components}, "
            f"scenarios={self.m_num_scenarios})"
        )
