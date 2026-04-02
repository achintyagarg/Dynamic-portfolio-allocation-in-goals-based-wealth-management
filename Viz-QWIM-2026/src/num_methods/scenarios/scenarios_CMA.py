r"""Scenarios from Capital Market Assumptions (CMA).

=====================================================

Provides :class:`Scenarios_CMA`, a concrete subclass of
:class:`Scenarios_Base` that generates daily financial time-series
scenarios from Capital Market Assumptions typically published by asset
managers and investment consultancies.

Capital Market Assumptions
--------------------------
A CMA data-set supplies, for each asset class, at least:

* expected annualised return
* expected annualised volatility (standard deviation)
* pairwise correlation matrix

This class translates those annual statistics into daily parameters and
either:

1. Reads a pre-built set of scenario paths from a spreadsheet, **or**
2. Generates scenarios internally from the CMA statistics assuming a
   correlated multivariate normal distribution of daily returns.

Asset-class tiers
-----------------
Asset classes are organised into three granularity tiers:

=====  =======================  =======================================
Tier   Example classes           Description
=====  =======================  =======================================
  0    Equities, Fixed Income,  Broadest allocation buckets
       Cash
  1    US Large Cap, US Mid     Sub-Asset classes within a Tier-0
       Cap, Int'l Developed     bucket
  2    US Large Cap Growth,     Style / factor splits within a Tier-1
       US Large Cap Value       class
=====  =======================  =======================================

Each asset class is also mapped to a representative financial index
(e.g. S&P 500 \u2194 US Large Cap Equity).

Author
------
QWIM Team

Version
-------
0.6.0 (2026-03-01)
"""

from __future__ import annotations

import datetime as dt
import math

from pathlib import Path
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

from .scenarios_base import (
    Frequency_Time_Series,
    Scenario_Data_Type,
    Scenarios_Base,
)


logger = get_logger(__name__)


# ======================================================================
# Enums
# ======================================================================


class Asset_Class_Tier(Enum):
    """Granularity tier for an asset class.

    Attributes
    ----------
    TIER_0 : int
        Broadest bucket (Equities, Fixed Income, Cash).
    TIER_1 : int
        Sub-asset classes (US Large Cap, Int'l Developed, etc.).
    TIER_2 : int
        Style / factor splits (US Large Cap Growth, US Large Cap Value).
    """

    TIER_0 = 0
    TIER_1 = 1
    TIER_2 = 2


class CMA_Source(Enum):
    """Origin of the Capital Market Assumptions data.

    Attributes
    ----------
    SPREADSHEET : str
        Loaded from an Excel / CSV file.
    MANUAL : str
        Provided programmatically (vectors & matrices).
    """

    SPREADSHEET = "Spreadsheet"
    MANUAL = "Manual"


# ======================================================================
# Default asset-class / index correspondence tables
# ======================================================================

#: Tier-0 broad asset classes
TIER_0_ASSET_CLASSES: list[str] = [
    "Equities",
    "Fixed Income",
    "Cash",
]

#: Tier-1 sub-asset classes
TIER_1_ASSET_CLASSES: list[str] = [
    "US Large Cap",
    "US Mid Cap",
    "US Small Cap",
    "International Developed",
    "Emerging Markets",
    "US Investment Grade Bonds",
    "US High Yield Bonds",
    "International Bonds",
    "US Treasury",
    "US TIPS",
    "Real Estate",
    "Commodities",
    "Cash Equivalents",
]

#: Tier-2 granular classes
TIER_2_ASSET_CLASSES: list[str] = [
    "US Large Cap Growth",
    "US Large Cap Value",
    "US Mid Cap Growth",
    "US Mid Cap Value",
    "US Small Cap Growth",
    "US Small Cap Value",
    "International Developed Large Cap",
    "International Developed Small Cap",
    "Emerging Markets Large Cap",
    "Emerging Markets Small Cap",
    "US Short-Term Bonds",
    "US Intermediate Bonds",
    "US Long-Term Bonds",
    "US Corporate Bonds",
    "US Municipal Bonds",
    "High-Yield Corporate Bonds",
    "Emerging Market Debt",
    "US REIT",
    "Global REIT",
    "Gold",
    "Broad Commodities",
    "Treasury Bills",
]

#: Mapping from asset class name → representative financial index.
DEFAULT_INDEX_MAP: dict[str, str] = {
    # --- Tier 0 --------------------------------------------------
    "Equities": "MSCI ACWI",
    "Fixed Income": "Bloomberg US Aggregate Bond",
    "Cash": "ICE BofA US 3-Month T-Bill",
    # --- Tier 1 --------------------------------------------------
    "US Large Cap": "S&P 500",
    "US Mid Cap": "S&P MidCap 400",
    "US Small Cap": "Russell 2000",
    "International Developed": "MSCI EAFE",
    "Emerging Markets": "MSCI Emerging Markets",
    "US Investment Grade Bonds": "Bloomberg US Aggregate Bond",
    "US High Yield Bonds": "ICE BofA US High Yield",
    "International Bonds": "Bloomberg Global Aggregate ex-US",
    "US Treasury": "Bloomberg US Treasury",
    "US TIPS": "Bloomberg US TIPS",
    "Real Estate": "FTSE Nareit All Equity REITs",
    "Commodities": "Bloomberg Commodity",
    "Cash Equivalents": "ICE BofA US 3-Month T-Bill",
    # --- Tier 2 --------------------------------------------------
    "US Large Cap Growth": "Russell 1000 Growth",
    "US Large Cap Value": "Russell 1000 Value",
    "US Mid Cap Growth": "Russell Midcap Growth",
    "US Mid Cap Value": "Russell Midcap Value",
    "US Small Cap Growth": "Russell 2000 Growth",
    "US Small Cap Value": "Russell 2000 Value",
    "International Developed Large Cap": "MSCI EAFE Large Cap",
    "International Developed Small Cap": "MSCI EAFE Small Cap",
    "Emerging Markets Large Cap": "MSCI EM Large Cap",
    "Emerging Markets Small Cap": "MSCI EM Small Cap",
    "US Short-Term Bonds": "Bloomberg US 1-3 Year Govt/Credit",
    "US Intermediate Bonds": "Bloomberg US Intermediate Govt/Credit",
    "US Long-Term Bonds": "Bloomberg US Long Govt/Credit",
    "US Corporate Bonds": "Bloomberg US Corporate",
    "US Municipal Bonds": "Bloomberg Municipal Bond",
    "High-Yield Corporate Bonds": "ICE BofA US High Yield",
    "Emerging Market Debt": "JPM EMBI Global Diversified",
    "US REIT": "FTSE Nareit All Equity REITs",
    "Global REIT": "FTSE EPRA Nareit Global",
    "Gold": "LBMA Gold Price PM",
    "Broad Commodities": "Bloomberg Commodity",
    "Treasury Bills": "ICE BofA US 3-Month T-Bill",
}

#: Tier membership: asset class → tier enum
DEFAULT_TIER_MAP: dict[str, Asset_Class_Tier] = {}
for _ac in TIER_0_ASSET_CLASSES:
    DEFAULT_TIER_MAP[_ac] = Asset_Class_Tier.TIER_0
for _ac in TIER_1_ASSET_CLASSES:
    DEFAULT_TIER_MAP[_ac] = Asset_Class_Tier.TIER_1
for _ac in TIER_2_ASSET_CLASSES:
    DEFAULT_TIER_MAP[_ac] = Asset_Class_Tier.TIER_2


# ======================================================================
# CMA Scenario class
# ======================================================================


class Scenarios_CMA(Scenarios_Base):
    """Scenarios driven by Capital Market Assumptions.

    Parameters
    ----------
    names_asset_classes : Sequence[str]
        Asset-class names that will serve as the component columns.
    expected_returns_annual : np.ndarray
        Annualised expected returns vector, shape ``(K,)``.
    expected_vols_annual : np.ndarray
        Annualised volatilities vector, shape ``(K,)``.
    correlation_matrix : np.ndarray
        ``(K, K)`` correlation matrix.
    dates : Sequence[dt.date] | None
        Pre-defined dates.  If *None*, dates are generated from
        *start_date* and *num_days*.
    start_date : dt.date | None
        First business day (used only when *dates* is None).
    num_days : int
        Number of trading days to generate (default 252 = 1 year).
    num_scenarios : int
        Number of independent scenario paths.
    data_type : Scenario_Data_Type
        Type of the generated data.
    frequency : Frequency_Time_Series
        Observation frequency (default DAILY).
    name_scenarios : str
        Human-readable label.
    index_map : dict[str, str] | None
        Asset-class → index mapping override.  Falls back to
        :data:`DEFAULT_INDEX_MAP`.
    tier_map : dict[str, Asset_Class_Tier] | None
        Asset-class → tier mapping override.  Falls back to
        :data:`DEFAULT_TIER_MAP`.
    source : CMA_Source
        Origin of the CMA data.
    random_seed : int | None
        Seed for reproducible generation (``None`` = random).

    Attributes
    ----------
    m_expected_returns_annual : np.ndarray
    m_expected_vols_annual : np.ndarray
    m_correlation_matrix : np.ndarray
    m_covariance_matrix_annual : np.ndarray
    m_index_map : dict[str, str]
    m_tier_map : dict[str, Asset_Class_Tier]
    m_source : CMA_Source
    m_random_seed : int | None
    m_start_date : dt.date
    m_num_days : int
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        names_asset_classes: Sequence[str],
        expected_returns_annual: np.ndarray,
        expected_vols_annual: np.ndarray,
        correlation_matrix: np.ndarray,
        dates: Sequence[dt.date] | None = None,
        start_date: dt.date | None = None,
        num_days: int = 252,
        num_scenarios: int = 1,
        data_type: Scenario_Data_Type = Scenario_Data_Type.RETURN_ARITHMETIC,
        frequency: Frequency_Time_Series = Frequency_Time_Series.DAILY,
        name_scenarios: str = "CMA Scenario",
        index_map: dict[str, str] | None = None,
        tier_map: dict[str, Asset_Class_Tier] | None = None,
        source: CMA_Source = CMA_Source.MANUAL,
        random_seed: int | None = None,
    ) -> None:
        K = len(names_asset_classes)

        # --- validate array shapes ---
        expected_returns_annual = np.asarray(
            expected_returns_annual,
            dtype=np.float64,
        )
        expected_vols_annual = np.asarray(
            expected_vols_annual,
            dtype=np.float64,
        )
        correlation_matrix = np.asarray(
            correlation_matrix,
            dtype=np.float64,
        )

        if expected_returns_annual.shape != (K,):
            raise Exception_Validation_Input(
                f"expected_returns_annual shape must be ({K},), "
                f"got {expected_returns_annual.shape}",
                field_name="expected_returns_annual",
                expected_type=f"({K},)",
                actual_value=expected_returns_annual.shape,
            )

        if expected_vols_annual.shape != (K,):
            raise Exception_Validation_Input(
                f"expected_vols_annual shape must be ({K},), got {expected_vols_annual.shape}",
                field_name="expected_vols_annual",
                expected_type=f"({K},)",
                actual_value=expected_vols_annual.shape,
            )

        if correlation_matrix.shape != (K, K):
            raise Exception_Validation_Input(
                f"correlation_matrix shape must be ({K}, {K}), got {correlation_matrix.shape}",
                field_name="correlation_matrix",
                expected_type=f"({K}, {K})",
                actual_value=correlation_matrix.shape,
            )

        # --- validate correlation matrix properties ---
        diag = np.diag(correlation_matrix)
        if not np.allclose(diag, 1.0, atol=1e-8):
            raise Exception_Validation_Input(
                "Diagonal of correlation matrix must be 1.0",
                field_name="correlation_matrix diagonal",
                expected_type="1.0",
                actual_value=diag.tolist(),
            )

        if not np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-8):
            raise Exception_Validation_Input(
                "Correlation matrix must be symmetric",
                field_name="correlation_matrix",
                expected_type="symmetric",
                actual_value="asymmetric",
            )

        # --- determine dates ---
        if dates is None and start_date is None:
            start_date = dt.datetime.now(tz=dt.UTC).date()

        if dates is None:
            dates = self._generate_business_dates(start_date, num_days)

        # --- base init ---
        super().__init__(
            names_components=list(names_asset_classes),
            dates=dates,
            data_type=data_type,
            frequency=frequency,
            num_scenarios=num_scenarios,
            name_scenarios=name_scenarios,
        )

        # --- CMA-specific members ---
        self.m_expected_returns_annual: np.ndarray = expected_returns_annual
        self.m_expected_vols_annual: np.ndarray = expected_vols_annual
        self.m_correlation_matrix: np.ndarray = correlation_matrix

        # Compute annual covariance: Cov = diag(sigma) @ Corr @ diag(sigma)
        D = np.diag(expected_vols_annual)
        self.m_covariance_matrix_annual: np.ndarray = D @ correlation_matrix @ D

        # --- mappings ---
        self.m_index_map: dict[str, str] = (
            dict(index_map)
            if index_map is not None
            else {ac: DEFAULT_INDEX_MAP.get(ac, "N/A") for ac in names_asset_classes}
        )
        self.m_tier_map: dict[str, Asset_Class_Tier] = (
            dict(tier_map)
            if tier_map is not None
            else {
                ac: DEFAULT_TIER_MAP.get(ac, Asset_Class_Tier.TIER_1) for ac in names_asset_classes
            }
        )

        self.m_source: CMA_Source = source
        self.m_random_seed: int | None = random_seed
        self.m_start_date: dt.date = (
            self.m_dates[0] if self.m_dates else dt.datetime.now(tz=dt.UTC).date()
        )
        self.m_num_days: int = num_days

        logger.info(
            "Scenarios_CMA created",
            extra={
                "event_type": "scenarios_cma_created",
                "name_scenarios": self.m_name_scenarios,
                "num_asset_classes": self.m_num_components,
                "num_dates": self.m_num_dates,
                "source": self.m_source.value,
            },
        )

    # ------------------------------------------------------------------
    # Properties (CMA-specific)
    # ------------------------------------------------------------------

    @property
    def expected_returns_annual(self) -> np.ndarray:
        """np.ndarray : Annualised expected returns (copy)."""
        return self.m_expected_returns_annual.copy()

    @property
    def expected_vols_annual(self) -> np.ndarray:
        """np.ndarray : Annualised volatilities (copy)."""
        return self.m_expected_vols_annual.copy()

    @property
    def correlation_matrix(self) -> np.ndarray:
        """np.ndarray : Correlation matrix (copy)."""
        return self.m_correlation_matrix.copy()

    @property
    def covariance_matrix_annual(self) -> np.ndarray:
        """np.ndarray : Annual covariance matrix (copy)."""
        return self.m_covariance_matrix_annual.copy()

    @property
    def index_map(self) -> dict[str, str]:
        """dict[str, str] : Asset-class \u2192 index mapping (copy)."""
        return dict(self.m_index_map)

    @property
    def tier_map(self) -> dict[str, Asset_Class_Tier]:
        """dict[str, Asset_Class_Tier] : Asset-class \u2192 tier mapping."""
        return dict(self.m_tier_map)

    @property
    def source(self) -> CMA_Source:
        """CMA_Source : Origin of the CMA data."""
        return self.m_source

    @property
    def random_seed(self) -> int | None:
        """Int | None : RNG seed for reproducibility."""
        return self.m_random_seed

    # ------------------------------------------------------------------
    # Table accessors
    # ------------------------------------------------------------------

    def get_index_correspondence_table(self) -> pl.DataFrame:
        """Build a Polars DataFrame of asset class \u2192 index correspondences.

        Returns
        -------
        pl.DataFrame
            Columns: ``asset_class``, ``financial_index``, ``tier``,
            ``expected_return``, ``expected_volatility``.
        """
        rows: list[dict] = []
        for idx_i, item_ac in enumerate(self.m_names_components):
            rows.append(
                {
                    "asset_class": item_ac,
                    "financial_index": self.m_index_map.get(item_ac, "N/A"),
                    "tier": self.m_tier_map.get(
                        item_ac,
                        Asset_Class_Tier.TIER_1,
                    ).name,
                    "expected_return": float(
                        self.m_expected_returns_annual[idx_i],
                    ),
                    "expected_volatility": float(
                        self.m_expected_vols_annual[idx_i],
                    ),
                },
            )
        return pl.DataFrame(rows)

    def get_asset_classes_by_tier(
        self,
        tier: Asset_Class_Tier,
    ) -> list[str]:
        """Return asset-class names belonging to a given tier.

        Parameters
        ----------
        tier : Asset_Class_Tier
            The tier to filter on.

        Returns
        -------
        list[str]
            Ordered list of matching asset-class names.
        """
        return [
            item_ac for item_ac in self.m_names_components if self.m_tier_map.get(item_ac) == tier
        ]

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self) -> pl.DataFrame:
        r"""Generate daily return scenarios from the CMA statistics.

        Daily parameters are derived from annual:

        $$
        \mu_{\text{daily}} = \frac{\mu_{\text{annual}}}{252},
        \qquad
        \Sigma_{\text{daily}} = \frac{\Sigma_{\text{annual}}}{252}
        $$

        Correlated daily returns are sampled from
        $\mathcal{N}(\mu_{\text{daily}},\, \Sigma_{\text{daily}})$
        using a Cholesky decomposition.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``Date`` column and one column per asset
            class, containing daily arithmetic returns.
        """
        rng = np.random.default_rng(self.m_random_seed)

        temp_dates = self.m_num_dates
        temp_num_components = self.m_num_components

        if temp_dates == 0:
            raise Exception_Calculation(
                "Cannot generate scenarios: no dates available",
            )

        # Convert annual to daily
        mu_daily = self.m_expected_returns_annual / self.m_frequency.value
        cov_daily = self.m_covariance_matrix_annual / self.m_frequency.value

        # Cholesky decomposition
        try:
            mat_chol = np.linalg.cholesky(cov_daily)
        except np.linalg.LinAlgError:
            # Fall back to eigenvalue correction for near-PSD matrices
            eigvals, eigvecs = np.linalg.eigh(cov_daily)
            eigvals = np.maximum(eigvals, 1e-4)  # Clip small/negative eigenvalues
            cov_daily = eigvecs @ np.diag(eigvals) @ eigvecs.T
            mat_chol = np.linalg.cholesky(cov_daily)
            logger.warning(
                "Covariance matrix was not positive-definite; eigenvalues were clipped to 1e-4",
            )

        # Sample: Z ~ N(0,I), then X = mu + L @ Z
        temp_RNG_normal = rng.standard_normal((temp_dates, temp_num_components))
        temp_returns = temp_RNG_normal @ mat_chol.T + mu_daily

        # Build DataFrame
        temp_data: dict[str, list] = {"Date": self.m_dates}
        for idx_j, item_name in enumerate(self.m_names_components):
            temp_data[item_name] = temp_returns[:, idx_j].tolist()

        self.m_df_scenarios = pl.DataFrame(temp_data).with_columns(
            pl.col("Date").cast(pl.Date),
        )

        # Cast numeric columns to Float64
        self.m_df_scenarios = self.m_df_scenarios.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False) for c in self.m_names_components],
        )

        logger.info(
            "Generated daily return scenarios",
            extra={
                "event_type": "scenarios_generated",
                "num_dates": temp_dates,
                "num_asset_classes": temp_num_components,
                "random_seed": self.m_random_seed,
            },
        )

        return self.m_df_scenarios

    # ------------------------------------------------------------------
    # Spreadsheet constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_spreadsheet(
        cls,
        path: str | Path,
        sheet_name_returns: str = "Expected Returns",
        sheet_name_volatilities: str = "Volatilities",
        sheet_name_correlations: str = "Correlations",
        sheet_name_scenarios: str | None = None,
        dates: Sequence[dt.date] | None = None,
        start_date: dt.date | None = None,
        num_days: int = 252,
        num_scenarios: int = 1,
        data_type: Scenario_Data_Type = Scenario_Data_Type.RETURN_ARITHMETIC,
        frequency: Frequency_Time_Series = Frequency_Time_Series.DAILY,
        name_scenarios: str = "CMA Spreadsheet",
        index_map: dict[str, str] | None = None,
        tier_map: dict[str, Asset_Class_Tier] | None = None,
        random_seed: int | None = None,
    ) -> Scenarios_CMA:
        """Construct a :class:`Scenarios_CMA` from an Excel workbook.

        The workbook must contain at least three sheets supplying the
        expected returns, volatilities, and correlation matrix.
        Optionally a fourth sheet can hold pre-built scenario paths.

        Parameters
        ----------
        path : str | Path
            Path to the ``.xlsx`` or ``.xls`` file.
        sheet_name_returns : str
            Sheet name containing annualised expected returns with
            columns ``asset_class`` and ``expected_return``.
        sheet_name_volatilities : str
            Sheet name containing annualised volatilities with columns
            ``asset_class`` and ``volatility``.
        sheet_name_correlations : str
            Sheet name containing the correlation matrix.  First column
            is ``asset_class``; remaining columns are the numeric
            correlations in the same order.
        sheet_name_scenarios : str | None
            Optional sheet with pre-generated scenario paths (columns
            ``Date``, then asset-class names).  When provided the
            :meth:`generate` method returns this data directly.
        dates, start_date, num_days, num_scenarios, data_type,
        frequency, name_scenarios, index_map, tier_map, random_seed
            Forwarded to the constructor.

        Returns
        -------
        Scenarios_CMA
            Fully configured instance.
        """
        path = Path(path)
        if not path.exists():
            raise Exception_Validation_Input(
                f"Spreadsheet not found: {path}",
                field_name="path",
                expected_type="existing file",
                actual_value=str(path),
            )

        # --- read returns ---
        df_ret = pl.read_excel(path, sheet_name=sheet_name_returns)
        asset_classes = df_ret["asset_class"].to_list()
        expected_returns = (
            df_ret["expected_return"]
            .to_numpy()
            .astype(
                np.float64,
            )
        )

        # --- read volatilities ---
        df_vol = pl.read_excel(path, sheet_name=sheet_name_volatilities)
        expected_vols = df_vol["volatility"].to_numpy().astype(np.float64)

        # --- read correlations ---
        df_corr = pl.read_excel(path, sheet_name=sheet_name_correlations)
        corr_cols = [c for c in df_corr.columns if c != "asset_class"]
        corr_matrix = df_corr.select(corr_cols).to_numpy().astype(np.float64)

        # --- construct instance ---
        instance = cls(
            names_asset_classes=asset_classes,
            expected_returns_annual=expected_returns,
            expected_vols_annual=expected_vols,
            correlation_matrix=corr_matrix,
            dates=dates,
            start_date=start_date,
            num_days=num_days,
            num_scenarios=num_scenarios,
            data_type=data_type,
            frequency=frequency,
            name_scenarios=name_scenarios,
            index_map=index_map,
            tier_map=tier_map,
            source=CMA_Source.SPREADSHEET,
            random_seed=random_seed,
        )

        # --- optionally load pre-built scenarios ---
        if sheet_name_scenarios is not None:
            df_scen = pl.read_excel(
                path,
                sheet_name=sheet_name_scenarios,
            )
            if "Date" in df_scen.columns:
                if df_scen.schema["Date"] == pl.Utf8:
                    df_scen = df_scen.with_columns(
                        pl.col("Date").str.strptime(
                            pl.Date,
                            format="%Y-%m-%d",
                        ),
                    )
                df_scen = df_scen.with_columns(
                    [
                        pl.col(c).cast(pl.Float64, strict=False)
                        for c in asset_classes
                        if c in df_scen.columns
                    ],
                )
                instance.m_df_scenarios = df_scen
                instance.m_dates = df_scen["Date"].to_list()
                instance.m_num_dates = len(instance.m_dates)
                logger.info(
                    "Loaded pre-built scenario rows from spreadsheet",
                    extra={
                        "event_type": "scenarios_loaded",
                        "num_dates": instance.m_num_dates,
                        "sheet_name": sheet_name_scenarios,
                    },
                )

        return instance

    # ------------------------------------------------------------------
    # Annualisation helpers
    # ------------------------------------------------------------------

    def calc_daily_expected_returns(self) -> np.ndarray:
        r"""Convert annual expected returns to daily.

        $$
        \mu_{\text{daily}} = \frac{\mu_{\text{annual}}}{N}
        $$

        where $N$ is the number of observations per year
        (``self.m_frequency.value``).

        Returns
        -------
        np.ndarray
            Daily expected returns, shape ``(K,)``.
        """
        return self.m_expected_returns_annual / self.m_frequency.value

    def calc_daily_covariance(self) -> np.ndarray:
        r"""Convert annual covariance matrix to daily.

        $$
        \Sigma_{\text{daily}} = \frac{\Sigma_{\text{annual}}}{N}
        $$

        Returns
        -------
        np.ndarray
            Daily covariance matrix, shape ``(K, K)``.
        """
        return self.m_covariance_matrix_annual / self.m_frequency.value

    def calc_daily_volatilities(self) -> np.ndarray:
        r"""Convert annual volatilities to daily.

        $$
        \sigma_{\text{daily}} = \frac{\sigma_{\text{annual}}}{\sqrt{N}}
        $$

        Returns
        -------
        np.ndarray
            Daily volatilities, shape ``(K,)``.
        """
        return self.m_expected_vols_annual / math.sqrt(self.m_frequency.value)

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Scenarios_CMA("
            f"name='{self.m_name_scenarios}', "
            f"asset_classes={self.m_num_components}, "
            f"dates={self.m_num_dates}, "
            f"source={self.m_source.value})"
        )
