r"""Base class for scenarios of daily financial time series.

===============================================================

Provides :class:`Scenarios_Base`, an abstract base class that stores
multi-component daily financial time series in a single
:class:`polars.DataFrame`.  The layout follows the convention::

    Date | Component_0 | Component_1 | \u2026 | Component_{K-1}

Each row represents one trading day and each numeric column holds the
value (price, return, or index level) of that component.

The class is designed for three primary use-cases:

1. **Monte Carlo simulation** \u2014 generate or load many scenario paths for
   portfolio stress-testing and retirement projections.
2. **Multi-component portfolio back-testing** \u2014 supply a date-aligned
   matrix of asset returns to the portfolio optimiser.
3. **Scenario-based optimisation** \u2014 feed structured or distributional
   scenarios into mean\u2013variance, CVaR or robust optimisers.

Design goals
------------
* **Performance** \u2014 all heavy-lifting delegated to Polars (columnar,
  SIMD, lazy evaluation).  Copies are avoided where possible.
* **Simplicity** \u2014 a thin, Pythonic wrapper with clear properties,
  validation, and a single canonical DataFrame.
* **Extensibility** \u2014 concrete subclasses only need to override
  :meth:`generate` (and optionally :meth:`validate_scenarios`).

Author
------
QWIM Team

Version
-------
0.6.0 (2026-03-01)
"""

from __future__ import annotations

import datetime as dt

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from aenum import Enum


if TYPE_CHECKING:
    from collections.abc import Sequence

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)


# ======================================================================
# Enums
# ======================================================================


class Scenario_Data_Type(Enum):
    r"""Type of data stored in the scenario DataFrame.

    Attributes
    ----------
    PRICE : str
        Absolute price levels.
    RETURN_ARITHMETIC : str
        Simple (arithmetic) returns  $r_t = P_t / P_{t-1} - 1$.
    RETURN_LOG : str
        Logarithmic returns  $r_t = \ln(P_t / P_{t-1})$.
    INDEX_LEVEL : str
        Rebased index (starting at 100 or 1).
    """

    PRICE = "Price"
    RETURN_ARITHMETIC = "Arithmetic Return"
    RETURN_LOG = "Log Return"
    INDEX_LEVEL = "Index Level"


class Frequency_Time_Series(Enum):
    """Observation frequency of the time series.

    Attributes
    ----------
    DAILY : int
        252 observations per year (trading days).
    WEEKLY : int
        52 observations per year.
    MONTHLY : int
        12 observations per year.
    QUARTERLY : int
        4 observations per year.
    ANNUAL : int
        1 observation per year.
    """

    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    QUARTERLY = 4
    ANNUAL = 1


# ======================================================================
# Base class
# ======================================================================


class Scenarios_Base(ABC):
    """Abstract base class for daily financial time-series scenarios.

    Stores a date-aligned matrix of scenario values in a single
    :class:`polars.DataFrame` with schema::

        {Date: pl.Date, comp_0: pl.Float64, comp_1: pl.Float64, \u2026}

    Subclasses **must** implement :meth:`generate`.

    Parameters
    ----------
    names_components : Sequence[str]
        Ordered names of the components (e.g. ``["US Equity", "US Bond"]``).
    dates : Sequence[dt.date] | None
        Vector of trading dates.  If *None* the subclass must supply dates
        via :meth:`generate`.
    data_type : Scenario_Data_Type
        Interpretation of the numeric columns (price, return, index).
    frequency : Frequency_Time_Series
        Observation frequency.  Defaults to ``DAILY``.
    num_scenarios : int
        Number of independent scenario paths.  Defaults to ``1``.
    name_scenarios : str
        Human-readable label for this scenario set.

    Attributes
    ----------
    m_names_components : list[str]
    m_dates : list[dt.date]
    m_data_type : Scenario_Data_Type
    m_frequency : Frequency_Time_Series
    m_num_scenarios : int
    m_name_scenarios : str
    m_num_components : int
    m_num_dates : int
    m_df_scenarios : pl.DataFrame | None
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        names_components: Sequence[str],
        dates: Sequence[dt.date] | None = None,
        data_type: Scenario_Data_Type = Scenario_Data_Type.RETURN_ARITHMETIC,
        frequency: Frequency_Time_Series = Frequency_Time_Series.DAILY,
        num_scenarios: int = 1,
        name_scenarios: str = "Scenario Set",
    ) -> None:
        # --- validate components ---
        if names_components is None or len(names_components) == 0:
            raise Exception_Validation_Input(
                "names_components must be a non-empty sequence of strings",
                field_name="names_components",
                expected_type="Sequence[str]",
                actual_value=names_components,
            )

        for idx, name in enumerate(names_components):
            if not isinstance(name, str) or len(name.strip()) == 0:
                raise Exception_Validation_Input(
                    f"Component name at index {idx} must be a non-empty string",
                    field_name=f"names_components[{idx}]",
                    expected_type=str,
                    actual_value=name,
                )

        if len(set(names_components)) != len(names_components):
            raise Exception_Validation_Input(
                "Component names must be unique",
                field_name="names_components",
                expected_type="unique strings",
                actual_value=names_components,
            )

        # --- validate enums ---
        if not isinstance(data_type, Scenario_Data_Type):
            raise Exception_Validation_Input(
                "data_type must be a Scenario_Data_Type enum member",
                field_name="data_type",
                expected_type=Scenario_Data_Type,
                actual_value=data_type,
            )

        if not isinstance(frequency, Frequency_Time_Series):
            raise Exception_Validation_Input(
                "frequency must be a Frequency_Time_Series enum member",
                field_name="frequency",
                expected_type=Frequency_Time_Series,
                actual_value=frequency,
            )

        # --- validate num_scenarios ---
        if not isinstance(num_scenarios, int) or num_scenarios < 1:
            raise Exception_Validation_Input(
                "num_scenarios must be a positive integer",
                field_name="num_scenarios",
                expected_type=int,
                actual_value=num_scenarios,
            )

        # --- store members ---
        self.m_names_components: list[str] = list(names_components)
        self.m_num_components: int = len(self.m_names_components)
        self.m_data_type: Scenario_Data_Type = data_type
        self.m_frequency: Frequency_Time_Series = frequency
        self.m_num_scenarios: int = num_scenarios
        self.m_name_scenarios: str = str(name_scenarios).strip()

        # --- dates ---
        if dates is not None:
            self.m_dates: list[dt.date] = self._parse_dates(dates)
            self.m_num_dates: int = len(self.m_dates)
        else:
            self.m_dates = []
            self.m_num_dates = 0

        # --- scenario data (populated by generate()) ---
        self.m_df_scenarios: pl.DataFrame | None = None

        logger.info(
            "Scenarios_Base created: '%s', %d components, %d dates, data_type=%s, frequency=%s",
            self.m_name_scenarios,
            self.m_num_components,
            self.m_num_dates,
            self.m_data_type.value,
            self.m_frequency.name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def names_components(self) -> list[str]:
        """list[str] : Ordered component names (copy)."""
        return self.m_names_components.copy()

    @property
    def num_components(self) -> int:
        """Int : Number of components."""
        return self.m_num_components

    @property
    def dates(self) -> list[dt.date]:
        """list[dt.date] : Ordered trading dates (copy)."""
        return self.m_dates.copy()

    @property
    def num_dates(self) -> int:
        """Int : Number of dates in the scenario."""
        return self.m_num_dates

    @property
    def data_type(self) -> Scenario_Data_Type:
        """Scenario_Data_Type : Kind of numeric data stored."""
        return self.m_data_type

    @property
    def frequency(self) -> Frequency_Time_Series:
        """Frequency_Time_Series : Observation frequency."""
        return self.m_frequency

    @property
    def num_scenarios(self) -> int:
        """Int : Number of independent scenario paths."""
        return self.m_num_scenarios

    @property
    def name_scenarios(self) -> str:
        """Str : Human-readable label."""
        return self.m_name_scenarios

    @property
    def df_scenarios(self) -> pl.DataFrame | None:
        """pl.DataFrame | None : The scenario data (None until generated)."""
        if self.m_df_scenarios is not None:
            return self.m_df_scenarios.clone()
        return None

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self) -> pl.DataFrame:
        """Generate or load scenario data and store in *m_df_scenarios*.

        Concrete subclasses must populate ``self.m_df_scenarios`` with a
        :class:`polars.DataFrame` whose first column is ``"Date"``
        (dtype ``pl.Date``) and subsequent columns correspond to
        ``m_names_components`` (dtype ``pl.Float64``).

        Returns
        -------
        pl.DataFrame
            The generated scenario DataFrame.
        """

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_scenarios(self) -> bool:
        """Validate the current scenario DataFrame.

        Checks
        ------
        1. ``m_df_scenarios`` is not None.
        2. First column is ``"Date"`` with a date-compatible dtype.
        3. Remaining columns match ``m_names_components`` in order.
        4. All numeric columns are ``pl.Float64``.
        5. No null / NaN values (unless overridden).

        Returns
        -------
        bool
            ``True`` if all checks pass.

        Raises
        ------
        Exception_Validation_Input
            When any check fails.
        """
        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario DataFrame has not been generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        df = self.m_df_scenarios

        # --- column names ---
        expected_cols = ["Date", *self.m_names_components]
        if list(df.columns) != expected_cols:
            raise Exception_Validation_Input(
                f"Column mismatch. Expected {expected_cols}, got {df.columns}",
                field_name="columns",
                expected_type=str,
                actual_value=df.columns,
            )

        # --- Date dtype ---
        date_dtype = df.schema["Date"]
        if date_dtype not in (pl.Date, pl.Datetime, pl.Utf8):
            raise Exception_Validation_Input(
                f"Date column dtype must be Date, Datetime, or Utf8; got {date_dtype}",
                field_name="Date dtype",
                expected_type="pl.Date",
                actual_value=str(date_dtype),
            )

        # --- numeric dtypes ---
        for col in self.m_names_components:
            if df.schema[col] != pl.Float64:
                raise Exception_Validation_Input(
                    f"Column '{col}' must be Float64, got {df.schema[col]}",
                    field_name=col,
                    expected_type=pl.Float64,
                    actual_value=str(df.schema[col]),
                )

        # --- nulls ---
        null_counts = df.null_count()
        for col in df.columns:
            n_nulls = null_counts[col][0]
            if n_nulls > 0:
                logger.warning(
                    "Column '%s' contains %d null values",
                    col,
                    n_nulls,
                )

        logger.info(
            "Scenario validation passed: %d rows x %d columns",
            df.shape[0],
            df.shape[1],
        )
        return True

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------

    def get_component_series(self, component_name: str) -> pl.Series:
        """Extract a single component's values as a Polars Series.

        Parameters
        ----------
        component_name : str
            Must be one of :attr:`names_components`.

        Returns
        -------
        pl.Series
            The float values for the requested component.

        Raises
        ------
        Exception_Validation_Input
            If the component name is not found or data not generated.
        """
        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        if component_name not in self.m_names_components:
            raise Exception_Validation_Input(
                f"Component '{component_name}' not in names_components",
                field_name="component_name",
                expected_type=str,
                actual_value=component_name,
            )

        return self.m_df_scenarios[component_name]

    def get_returns_matrix(self) -> np.ndarray:
        """Return the numeric columns as a NumPy array.

        Returns a ``(T, K)`` matrix where *T* = number of dates and
        *K* = number of components.  Only the numeric columns are
        included (Date is excluded).

        Returns
        -------
        np.ndarray
            Array of shape ``(T, K)`` with dtype ``float64``.

        Raises
        ------
        Exception_Validation_Input
            If the scenario data has not been generated.
        """
        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        return self.m_df_scenarios.select(self.m_names_components).to_numpy()

    def get_date_range(self) -> tuple[dt.date, dt.date]:
        """Return the first and last date in the scenario.

        Returns
        -------
        tuple[dt.date, dt.date]
            ``(start_date, end_date)``.

        Raises
        ------
        Exception_Validation_Input
            If dates are empty.
        """
        if self.m_num_dates == 0:
            raise Exception_Validation_Input(
                "No dates available",
                field_name="m_dates",
                expected_type="non-empty list",
                actual_value=[],
            )
        return self.m_dates[0], self.m_dates[-1]

    def filter_by_date_range(
        self,
        start_date: dt.date,
        end_date: dt.date,
    ) -> pl.DataFrame:
        """Return a filtered copy of the scenario DataFrame.

        Parameters
        ----------
        start_date : dt.date
            Inclusive start date.
        end_date : dt.date
            Inclusive end date.

        Returns
        -------
        pl.DataFrame
            Filtered DataFrame containing only rows within the range.

        Raises
        ------
        Exception_Validation_Input
            If the scenario data is not available.
        """
        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        return self.m_df_scenarios.filter(
            (pl.col("Date") >= start_date) & (pl.col("Date") <= end_date),
        )

    def select_components(
        self,
        component_names: Sequence[str],
    ) -> pl.DataFrame:
        """Return a DataFrame with only the requested components.

        Parameters
        ----------
        component_names : Sequence[str]
            Subset of :attr:`names_components` to keep.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``Date`` + selected component columns.
        """
        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        for name in component_names:
            if name not in self.m_names_components:
                raise Exception_Validation_Input(
                    f"Component '{name}' not in scenario components",
                    field_name="component_names",
                    expected_type=str,
                    actual_value=name,
                )

        return self.m_df_scenarios.select(["Date", *component_names])

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def convert_prices_to_returns(
        self,
        log_returns: bool = False,
    ) -> pl.DataFrame:
        r"""Convert price-level scenarios to return scenarios.

        Computes arithmetic or log returns from consecutive prices:

        $$
        r_t^{\text{arith}} = \frac{P_t}{P_{t-1}} - 1
        \qquad
        r_t^{\log} = \ln\!\left(\frac{P_t}{P_{t-1}}\right)
        $$

        Parameters
        ----------
        log_returns : bool
            If *True* compute log returns; otherwise arithmetic.

        Returns
        -------
        pl.DataFrame
            DataFrame of returns (one row shorter than the price data).

        Raises
        ------
        Exception_Calculation
            If the current data type is not ``PRICE`` or
            ``INDEX_LEVEL``.
        """
        if self.m_data_type not in (
            Scenario_Data_Type.PRICE,
            Scenario_Data_Type.INDEX_LEVEL,
        ):
            raise Exception_Calculation(
                "convert_prices_to_returns requires PRICE or INDEX_LEVEL "
                f"data, got {self.m_data_type.value}",
            )

        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        df = self.m_df_scenarios
        cols = self.m_names_components

        if log_returns:
            return_exprs = [(pl.col(c) / pl.col(c).shift(1)).log().alias(c) for c in cols]
        else:
            return_exprs = [(pl.col(c) / pl.col(c).shift(1) - 1.0).alias(c) for c in cols]

        return (
            df.with_columns(return_exprs).slice(1)  # drop first row (NaN from shift)
        )

    def convert_returns_to_prices(
        self,
        initial_prices: Sequence[float] | None = None,
    ) -> pl.DataFrame:
        r"""Convert return scenarios to price levels via cumulative product.

        $$
        P_t = P_0 \prod_{s=1}^{t} (1 + r_s)
        $$

        Parameters
        ----------
        initial_prices : Sequence[float] | None
            Starting price per component.  Defaults to 100.0 for each.

        Returns
        -------
        pl.DataFrame
            Price-level DataFrame.

        Raises
        ------
        Exception_Calculation
            If the current data type is not a return type.
        """
        if self.m_data_type not in (
            Scenario_Data_Type.RETURN_ARITHMETIC,
            Scenario_Data_Type.RETURN_LOG,
        ):
            raise Exception_Calculation(
                "convert_returns_to_prices requires RETURN_ARITHMETIC or "
                f"RETURN_LOG data, got {self.m_data_type.value}",
            )

        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        if initial_prices is None:
            initial_prices = [100.0] * self.m_num_components

        if len(initial_prices) != self.m_num_components:
            raise Exception_Validation_Input(
                f"initial_prices length ({len(initial_prices)}) must match "
                f"num_components ({self.m_num_components})",
                field_name="initial_prices",
                expected_type=f"len == {self.m_num_components}",
                actual_value=len(initial_prices),
            )

        df = self.m_df_scenarios
        cols = self.m_names_components

        if self.m_data_type == Scenario_Data_Type.RETURN_LOG:
            price_exprs = [
                (pl.col(c).cum_sum().exp() * p0).alias(c)
                for c, p0 in zip(cols, initial_prices, strict=True)
            ]
        else:
            price_exprs = [
                ((1.0 + pl.col(c)).cum_prod() * p0).alias(c)
                for c, p0 in zip(cols, initial_prices, strict=True)
            ]

        return df.with_columns(price_exprs)

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def calc_summary_statistics(self) -> pl.DataFrame:
        """Compute per-component summary statistics.

        Returns a DataFrame with one row per component and columns:
        ``component``, ``mean``, ``std``, ``min``, ``max``,
        ``median``, ``skew``, ``count``.

        Returns
        -------
        pl.DataFrame
            Summary statistics.
        """
        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        records: list[dict] = []
        for idx_col in self.m_names_components:
            s = self.m_df_scenarios[idx_col]
            records.append(
                {
                    "component": idx_col,
                    "mean": s.mean(),
                    "std": s.std(),
                    "min": s.min(),
                    "max": s.max(),
                    "median": s.median(),
                    "skew": s.skew(),
                    "count": s.len(),
                },
            )

        return pl.DataFrame(records)

    def calc_correlation_matrix(self) -> pl.DataFrame:
        """Compute the pairwise correlation matrix across components.

        Returns
        -------
        pl.DataFrame
            ``K x K`` correlation matrix with component names as both
            the row identifier column (``"component"``) and column names.
        """
        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        mat = self.get_returns_matrix()
        corr = np.corrcoef(mat, rowvar=False)

        rows: list[dict] = []
        for idx_i, name_i in enumerate(self.m_names_components):
            row = {"component": name_i}
            for idx_j, name_j in enumerate(self.m_names_components):
                row[name_j] = float(corr[idx_i, idx_j])
            rows.append(row)

        return pl.DataFrame(rows)

    def calc_covariance_matrix(self) -> pl.DataFrame:
        """Compute the pairwise covariance matrix across components.

        Returns
        -------
        pl.DataFrame
            ``K x K`` covariance matrix with component names as both
            the row identifier column (``"component"``) and column names.
        """
        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )

        mat = self.get_returns_matrix()
        cov = np.cov(mat, rowvar=False)

        rows: list[dict] = []
        for idx_i, name_i in enumerate(self.m_names_components):
            row = {"component": name_i}
            for idx_j, name_j in enumerate(self.m_names_components):
                row[name_j] = float(cov[idx_i, idx_j])
            rows.append(row)

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def to_csv(self, path: str) -> None:
        """Write the scenario DataFrame to a CSV file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        if self.m_df_scenarios is None:
            raise Exception_Validation_Input(
                "Scenario data not generated yet",
                field_name="m_df_scenarios",
                expected_type=pl.DataFrame,
                actual_value=None,
            )
        self.m_df_scenarios.write_csv(path)
        logger.info(
            "Scenarios written to file",
            extra={"file_path": path},
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        data_type: Scenario_Data_Type = Scenario_Data_Type.RETURN_ARITHMETIC,
        frequency: Frequency_Time_Series = Frequency_Time_Series.DAILY,
        name_scenarios: str = "Loaded Scenario",
    ) -> Scenarios_Base:
        """Load scenarios from a CSV file.

        The CSV must have a ``Date`` column followed by one or more
        numeric columns.

        Parameters
        ----------
        path : str
            Path to CSV file.
        data_type : Scenario_Data_Type
            How to interpret the numeric values.
        frequency : Frequency_Time_Series
            Observation frequency.
        name_scenarios : str
            Label for the loaded set.

        Returns
        -------
        Scenarios_Base
            A concrete instance with pre-loaded data.

        Notes
        -----
        Because :class:`Scenarios_Base` is abstract this classmethod
        delegates to a lightweight inner subclass that satisfies the
        ABC contract.
        """
        df = pl.read_csv(path)

        if "Date" not in df.columns:
            raise Exception_Validation_Input(
                "CSV must contain a 'Date' column",
                field_name="Date",
                expected_type="column",
                actual_value=df.columns,
            )

        # Cast Date
        if df.schema["Date"] == pl.Utf8:
            df = df.with_columns(
                pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d"),
            )

        component_cols = [c for c in df.columns if c != "Date"]

        # Cast numerics
        df = df.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False) for c in component_cols],
        )

        dates = df["Date"].to_list()

        class _Loaded_Scenarios(Scenarios_Base):
            """Thin concrete wrapper for CSV-loaded data."""

            def generate(self) -> pl.DataFrame:
                return self.m_df_scenarios

        instance = _Loaded_Scenarios(
            names_components=component_cols,
            dates=dates,
            data_type=data_type,
            frequency=frequency,
            name_scenarios=name_scenarios,
        )
        instance.m_df_scenarios = df
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dates(dates: Sequence) -> list[dt.date]:
        """Coerce a date-like sequence into ``list[datetime.date]``.

        Accepts ``datetime.date``, ``datetime.datetime``, or ISO-8601
        strings (``YYYY-MM-DD``).

        Parameters
        ----------
        dates : Sequence
            Date-like objects.

        Returns
        -------
        list[dt.date]
            Parsed and sorted dates.

        Raises
        ------
        Exception_Validation_Input
            If any element cannot be parsed.
        """
        parsed: list[dt.date] = []
        for idx, d in enumerate(dates):
            if isinstance(d, dt.datetime):
                parsed.append(d.date())
            elif isinstance(d, dt.date):
                parsed.append(d)
            elif isinstance(d, str):
                try:
                    parsed.append(dt.date.fromisoformat(d))
                except ValueError as exc:
                    raise Exception_Validation_Input(
                        f"Cannot parse date string at index {idx}: '{d}'",
                        field_name=f"dates[{idx}]",
                        expected_type="ISO-8601 string",
                        actual_value=d,
                    ) from exc
            else:
                raise Exception_Validation_Input(
                    f"Unsupported date type at index {idx}: {type(d)}",
                    field_name=f"dates[{idx}]",
                    expected_type="date | datetime | str",
                    actual_value=type(d).__name__,
                )

        parsed.sort()
        return parsed

    @staticmethod
    def _generate_business_dates(
        start_date: dt.date,
        num_days: int,
    ) -> list[dt.date]:
        """Generate *num_days* business dates starting from *start_date*.

        Skips Saturday (5) and Sunday (6).

        Parameters
        ----------
        start_date : dt.date
            First date in the sequence.
        num_days : int
            Number of business days to generate.

        Returns
        -------
        list[dt.date]
            Ordered business day dates.
        """
        result: list[dt.date] = []
        current = start_date
        while len(result) < num_days:
            if current.weekday() < 5:  # Mon-Fri
                result.append(current)
            current += dt.timedelta(days=1)
        return result

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.m_name_scenarios}', "
            f"components={self.m_num_components}, "
            f"dates={self.m_num_dates}, "
            f"data_type={self.m_data_type.value})"
        )

    def __len__(self) -> int:
        """Return the number of dates (rows)."""
        return self.m_num_dates
