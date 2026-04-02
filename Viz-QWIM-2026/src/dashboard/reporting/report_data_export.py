"""Data Export Module for QWIM Report Generation.

Exports dashboard data (client info, portfolio inputs/outputs, simulation
inputs/outputs) to JSON files that the Typst template reads directly via
``json()`` calls. This replaces the legacy ``{{PLACEHOLDER}}`` substitution
approach.

Each public function writes a single JSON file into the appropriate
subdirectory beneath ``src/dashboard/reporting/``:

* ``client_info.json`` — investor personal info, assets, goals, income
* ``inputs_json/*.json`` — user-selected inputs per subtab
* ``outputs_json/*.json`` — computed outputs / metrics per subtab

Dependencies
------------
* polars: DataFrame handling
* json: Serialisation

Notes
-----
All monetary values are stored as floats (not ``Decimal``) in JSON because
Typst does not natively parse ``Decimal``.  Formatting to currency strings
is done inside the Typst template.
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPORTING_DIR = Path(__file__).resolve().parent
_INPUTS_JSON_DIR = _REPORTING_DIR / "inputs_json"
_OUTPUTS_JSON_DIR = _REPORTING_DIR / "outputs_json"

# Project-root paths for sample CSV fallback data
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROCESSED_DATA_DIR = _PROJECT_ROOT / "inputs" / "processed"
_SAMPLE_PORTFOLIO_CSV = _PROCESSED_DATA_DIR / "sample_portfolio_values.csv"
_BENCHMARK_PORTFOLIO_CSV = _PROCESSED_DATA_DIR / "benchmark_portfolio_values.csv"

# Financial constants used in metric calculations
_TRADING_DAYS_PER_YEAR: float = 252.0
_RISK_FREE_RATE_ANNUAL: float = 0.02  # 2 % nominal risk-free rate

# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------


def validate_report_data_quality() -> dict[str, list[str]]:
    """Check exported JSON files for data quality issues.

    Reads each JSON file in the inputs/outputs directories and checks for:

    * Missing files
    * Empty arrays where data is expected (e.g. ``weight_statistics == []``)
    * All-zero numeric fields in output tables

    Returns
    -------
    dict[str, list[str]]
        Mapping of section name to list of issue descriptions.
        An empty dict means all data is healthy.
    """
    issues: dict[str, list[str]] = {}

    def _check_json(path: Path, section_name: str, array_keys: list[str] | None = None) -> None:
        if not path.exists():
            issues.setdefault(section_name, []).append(f"JSON file missing: {path.name}")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.setdefault(section_name, []).append(f"Cannot parse {path.name}: {exc}")
            return
        if not isinstance(data, dict) or not data:
            issues.setdefault(section_name, []).append(
                f"{path.name} is empty or not a valid object",
            )
            return
        if array_keys:
            for key in array_keys:
                arr = data.get(key)
                if not isinstance(arr, list) or len(arr) == 0:
                    issues.setdefault(section_name, []).append(
                        f"Table '{key}' in {path.name} is empty — report will show 'No data available'",
                    )

    # Report metadata & client info
    _check_json(_REPORTING_DIR / "report_metadata.json", "Report Metadata")
    _check_json(_REPORTING_DIR / "client_info.json", "Client Information")

    # Outputs with tables
    _check_json(
        _OUTPUTS_JSON_DIR / "outputs_weights_analysis.json",
        "Weights Analysis",
        array_keys=["weight_statistics"],
    )
    _check_json(
        _OUTPUTS_JSON_DIR / "outputs_portfolio_analysis.json",
        "Portfolio Analysis",
        array_keys=["basic_statistics", "performance_metrics"],
    )
    _check_json(
        _OUTPUTS_JSON_DIR / "outputs_skfolio_optimization.json",
        "Skfolio Optimization",
        array_keys=["weights_comparison", "statistics_comparison"],
    )
    # Simulation — check summary_statistics for all-zero
    sim_path = _OUTPUTS_JSON_DIR / "outputs_simulation.json"
    if sim_path.exists():
        try:
            sim_data = json.loads(sim_path.read_text(encoding="utf-8"))
            ss = sim_data.get("summary_statistics", {})
            if isinstance(ss, dict):
                numeric_vals = [
                    v for k, v in ss.items() if isinstance(v, (int, float)) and k != "initial_value"
                ]
                if numeric_vals and all(v == 0 or v == 0.0 for v in numeric_vals):
                    issues.setdefault("Simulation", []).append(
                        "All simulation statistics are zero — simulation may not have been run",
                    )
        except Exception:
            pass

    return issues


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dir(directory: Path) -> None:
    """Create *directory* if it does not already exist."""
    directory.mkdir(parents=True, exist_ok=True)


def _write_json(file_path: Path, data: dict[str, Any]) -> None:
    """Serialise *data* to *file_path* as pretty-printed JSON."""
    _ensure_dir(file_path.parent)
    file_path.write_text(
        json.dumps(data, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote JSON: %s (%d bytes)", file_path.name, file_path.stat().st_size)


def _safe_reactive_get(reactive_value: Any) -> Any:
    """Unwrap a Shiny ``reactive.Value`` or return the raw value."""
    if reactive_value is None:
        return None
    if hasattr(reactive_value, "get"):
        try:
            return reactive_value.get()
        except Exception:
            return None
    return reactive_value


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert *value* to ``float``, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_str(value: Any, default: str = "N/A") -> str:
    """Convert *value* to ``str``, returning *default* on failure."""
    if value is None:
        return default
    return str(value)


def _polars_to_records(df: pl.DataFrame | None) -> list[dict[str, Any]]:
    """Convert a Polars DataFrame to a list of row-dicts suitable for JSON."""
    if df is None or not isinstance(df, pl.DataFrame) or df.height == 0:
        return []
    return df.to_dicts()


# ---------------------------------------------------------------------------
# Sample-data helpers (used when live dashboard data is unavailable)
# ---------------------------------------------------------------------------


def _load_sample_portfolio_data() -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load sample portfolio and benchmark value CSVs from *inputs/processed/*.

    Returns
    -------
    tuple[pl.DataFrame | None, pl.DataFrame | None]
        ``(portfolio_values_df, benchmark_values_df)`` — either may be
        ``None`` when the file cannot be read.
    """

    def _load(path: Path) -> pl.DataFrame | None:
        try:
            df = pl.read_csv(path)
            # Dates stored as timezone-aware strings: "2012-01-03 00:00:00-05:00"
            df = df.with_columns(
                pl.col("Date").str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d").alias("Date"),
            )
            df = df.rename({"Value": "portfolio_value"})
            return df.sort("Date")
        except Exception as exc:
            logger.debug("Could not load sample data from %s: %s", path, exc)
            return None

    return _load(_SAMPLE_PORTFOLIO_CSV), _load(_BENCHMARK_PORTFOLIO_CSV)


def _compute_portfolio_metrics_from_values(
    pv: pl.DataFrame,
    bv: pl.DataFrame | None = None,
    time_period: str = "All",
) -> dict[str, Any]:
    """Compute portfolio-comparison metrics from Polars DataFrames.

    Parameters
    ----------
    pv : pl.DataFrame
        Portfolio value series – must have ``Date`` and ``portfolio_value`` columns.
    bv : pl.DataFrame | None
        Benchmark value series in the same format.
    time_period : str
        Display label for the analysis period.

    Returns
    -------
    dict[str, Any]
        Flat dictionary of computed metrics (same keys as ``Portfolio_Comparison_Stats``).
    """
    pv_arr = pv["portfolio_value"].to_numpy()
    p_rets = np.diff(pv_arr) / pv_arr[:-1]
    n_days = len(pv_arr)
    n_years = n_days / _TRADING_DAYS_PER_YEAR

    total_return = float((pv_arr[-1] - pv_arr[0]) / pv_arr[0])
    ann_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    p_std = float(np.std(p_rets, ddof=1))
    ann_vol = p_std * float(np.sqrt(_TRADING_DAYS_PER_YEAR))

    cum_max = np.maximum.accumulate(pv_arr)
    max_drawdown = float(np.min((pv_arr - cum_max) / cum_max))

    rf_daily = _RISK_FREE_RATE_ANNUAL / _TRADING_DAYS_PER_YEAR
    excess = p_rets - rf_daily
    excess_std = float(np.std(excess, ddof=1))
    sharpe = (
        float(np.mean(excess)) / excess_std * float(np.sqrt(_TRADING_DAYS_PER_YEAR))
        if excess_std > 0
        else 0.0
    )

    start_date = str(pv["Date"][0])
    end_date = str(pv["Date"][-1])

    result: dict[str, Any] = {
        "time_period": time_period,
        "start_date": start_date,
        "end_date": end_date,
        "viz_type": "normalized",
        "total_return_portfolio": total_return,
        "annualized_return_portfolio": ann_return,
        "volatility_portfolio": ann_vol,
        "max_drawdown_portfolio": max_drawdown,
        "sharpe_ratio_portfolio": sharpe,
        "total_return_benchmark": 0.0,
        "annualized_return_benchmark": 0.0,
        "volatility_benchmark": 0.0,
        "max_drawdown_benchmark": 0.0,
        "sharpe_ratio_benchmark": 0.0,
        "total_return_difference": total_return,
        "annualized_return_difference": ann_return,
        "volatility_difference": ann_vol,
        "max_drawdown_difference": max_drawdown,
        "sharpe_ratio_difference": sharpe,
        "correlation": 0.0,
        "tracking_error": 0.0,
        "information_ratio": 0.0,
    }

    if bv is not None and bv.height >= 2:
        bv_arr = bv["portfolio_value"].to_numpy()
        b_rets = np.diff(bv_arr) / bv_arr[:-1]

        b_total = float((bv_arr[-1] - bv_arr[0]) / bv_arr[0])
        b_ann = float((1 + b_total) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        b_std = float(np.std(b_rets, ddof=1))
        b_vol = b_std * float(np.sqrt(_TRADING_DAYS_PER_YEAR))

        b_cum_max = np.maximum.accumulate(bv_arr)
        b_max_drawdown = float(np.min((bv_arr - b_cum_max) / b_cum_max))

        b_excess = b_rets - rf_daily
        b_excess_std = float(np.std(b_excess, ddof=1))
        b_sharpe = (
            float(np.mean(b_excess)) / b_excess_std * float(np.sqrt(_TRADING_DAYS_PER_YEAR))
            if b_excess_std > 0
            else 0.0
        )

        min_len = min(len(p_rets), len(b_rets))
        if min_len > 1:
            correlation = float(np.corrcoef(p_rets[:min_len], b_rets[:min_len])[0, 1])
            active = p_rets[:min_len] - b_rets[:min_len]
            te_daily = float(np.std(active, ddof=1))
            tracking_error = te_daily * float(np.sqrt(_TRADING_DAYS_PER_YEAR))
            info_ratio = (ann_return - b_ann) / tracking_error if tracking_error > 0 else 0.0
        else:
            correlation = 0.0
            tracking_error = 0.0
            info_ratio = 0.0

        result.update(
            {
                "total_return_benchmark": b_total,
                "annualized_return_benchmark": b_ann,
                "volatility_benchmark": b_vol,
                "max_drawdown_benchmark": b_max_drawdown,
                "sharpe_ratio_benchmark": b_sharpe,
                "total_return_difference": total_return - b_total,
                "annualized_return_difference": ann_return - b_ann,
                "volatility_difference": ann_vol - b_vol,
                "max_drawdown_difference": max_drawdown - b_max_drawdown,
                "sharpe_ratio_difference": sharpe - b_sharpe,
                "correlation": correlation,
                "tracking_error": tracking_error,
                "information_ratio": info_ratio,
            },
        )

    return result


def _compute_portfolio_analysis_stats_from_values(
    pv: pl.DataFrame,
    bv: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute basic statistics and risk-adjusted performance metrics.

    Parameters
    ----------
    pv : pl.DataFrame
        Portfolio value series.
    bv : pl.DataFrame | None
        Optional benchmark value series.

    Returns
    -------
    dict[str, Any]
        Dict with ``basic_statistics`` and ``performance_metrics`` lists.
    """
    pv_arr = pv["portfolio_value"].to_numpy()
    p_rets = np.diff(pv_arr) / pv_arr[:-1]
    n = len(p_rets)
    mean_d = float(np.mean(p_rets))
    std_d = float(np.std(p_rets, ddof=1))

    skewness = float(np.mean(((p_rets - mean_d) / std_d) ** 3)) if n > 2 and std_d > 0 else 0.0
    kurtosis = (
        float(np.mean(((p_rets - mean_d) / std_d) ** 4)) - 3.0 if n > 3 and std_d > 0 else 0.0
    )

    bm_mean = bm_std = bm_skew = bm_kurt = ""
    if bv is not None and bv.height >= 2:
        bv_arr = bv["portfolio_value"].to_numpy()
        b_rets = np.diff(bv_arr) / bv_arr[:-1]
        bm_mean_d = float(np.mean(b_rets))
        bm_std_d = float(np.std(b_rets, ddof=1))
        nb = len(b_rets)
        bm_mean = f"{bm_mean_d * 100:.4f}%"
        bm_std = f"{bm_std_d * 100:.4f}%"
        bm_skew = (
            f"{float(np.mean(((b_rets - bm_mean_d) / bm_std_d) ** 3)):.4f}"
            if nb > 2 and bm_std_d > 0
            else "0.0000"
        )
        bm_kurt = (
            f"{float(np.mean(((b_rets - bm_mean_d) / bm_std_d) ** 4)) - 3.0:.4f}"
            if nb > 3 and bm_std_d > 0
            else "0.0000"
        )

    basic_statistics = [
        {"metric": "Mean Daily Return", "portfolio": f"{mean_d * 100:.4f}%", "benchmark": bm_mean},
        {"metric": "Std Dev Daily Return", "portfolio": f"{std_d * 100:.4f}%", "benchmark": bm_std},
        {"metric": "Skewness", "portfolio": f"{skewness:.4f}", "benchmark": bm_skew},
        {"metric": "Excess Kurtosis", "portfolio": f"{kurtosis:.4f}", "benchmark": bm_kurt},
    ]

    total_return = float((pv_arr[-1] - pv_arr[0]) / pv_arr[0])
    n_years = len(pv_arr) / _TRADING_DAYS_PER_YEAR
    ann_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    ann_vol = std_d * float(np.sqrt(_TRADING_DAYS_PER_YEAR))
    rf_daily = _RISK_FREE_RATE_ANNUAL / _TRADING_DAYS_PER_YEAR
    excess = p_rets - rf_daily
    exc_std = float(np.std(excess, ddof=1))
    sharpe = (
        float(np.mean(excess)) / exc_std * float(np.sqrt(_TRADING_DAYS_PER_YEAR))
        if exc_std > 0
        else 0.0
    )
    cum_max = np.maximum.accumulate(pv_arr)
    max_drawdown = float(np.min((pv_arr - cum_max) / cum_max))

    performance_metrics = [
        {"metric": "Annualised Return", "value": f"{ann_return * 100:.2f}%"},
        {"metric": "Annualised Volatility", "value": f"{ann_vol * 100:.2f}%"},
        {"metric": "Sharpe Ratio (2 % Rf)", "value": f"{sharpe:.4f}"},
        {"metric": "Maximum Drawdown", "value": f"{max_drawdown * 100:.2f}%"},
        {"metric": "Analysis Period (yrs)", "value": f"{n_years:.1f}"},
    ]

    return {"basic_statistics": basic_statistics, "performance_metrics": performance_metrics}


def _compute_simulation_stats_from_values(
    pv: pl.DataFrame,
    num_scenarios: int = 1000,
    num_days: int = 252,
    initial_value: float = 100.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a simple normal Monte Carlo simulation calibrated to historical returns.

    Parameters
    ----------
    pv : pl.DataFrame
        Portfolio value series used to calibrate mean / std of returns.
    num_scenarios : int
        Number of simulation paths.
    num_days : int
        Simulation horizon in trading days.
    initial_value : float
        Starting portfolio value.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Summary statistics dict matching the ``Simulation_Stats`` schema.
    """
    pv_arr = pv["portfolio_value"].to_numpy()
    rets = np.diff(pv_arr) / pv_arr[:-1]
    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1))

    rng = np.random.default_rng(seed)
    sim_rets = rng.normal(mu, sigma, size=(num_scenarios, num_days))
    terminal_vals = initial_value * np.prod(1.0 + sim_rets, axis=1)

    return {
        "num_scenarios": num_scenarios,
        "horizon_days": num_days,
        "initial_value": initial_value,
        "mean_terminal_value": float(np.mean(terminal_vals)),
        "median_terminal_value": float(np.median(terminal_vals)),
        "std_dev_terminal_value": float(np.std(terminal_vals, ddof=1)),
        "percentile_5": float(np.percentile(terminal_vals, 5)),
        "percentile_25": float(np.percentile(terminal_vals, 25)),
        "percentile_75": float(np.percentile(terminal_vals, 75)),
        "percentile_95": float(np.percentile(terminal_vals, 95)),
        "min_terminal_value": float(np.min(terminal_vals)),
        "max_terminal_value": float(np.max(terminal_vals)),
        "probability_of_loss": float(np.mean(terminal_vals < initial_value)),
    }


# =========================================================================
# 0.  REPORT METADATA
# =========================================================================


def export_report_metadata(
    reactives_shiny: dict | None = None,  # noqa: ARG001
) -> Path:
    """Export report metadata (date, version) to ``report_metadata.json``.

    Parameters
    ----------
    reactives_shiny : dict | None
        Unused — present for API consistency.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    import datetime

    today = datetime.date.today()
    data: dict[str, str] = {
        "report_date": today.strftime("%B %d, %Y"),
        "report_date_iso": today.isoformat(),
        "report_version": "1.0",
        "generated_by": "QWIM Analytics Platform",
    }
    out_path = _REPORTING_DIR / "report_metadata.json"
    _write_json(out_path, data)
    return out_path


# =========================================================================
# 1.  CLIENT INFO
# =========================================================================


def export_client_info(
    reactives_shiny: dict | None,
) -> Path:
    """Export investor data to ``client_info.json``.

    Reads client information stored in ``reactives_shiny["Data_Clients"]`` via
    :func:`~src.dashboard.shiny_utils.utils_reporting.build_client_info_json_from_reactives`
    and writes the resulting dictionary to ``client_info.json`` in the reporting
    directory.  When the ``Data_Clients`` structure is absent or unpopulated, an
    empty ``{}`` fallback is written so that downstream Typst compilation still
    succeeds.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary containing ``Data_Clients``.

    Returns
    -------
    Path
        Absolute path to the written JSON file.
    """
    from src.dashboard.shiny_utils.utils_reporting import build_client_info_json_from_reactives

    data: dict[str, Any] = {}
    try:
        data = build_client_info_json_from_reactives(reactives_shiny)
        logger.info("Built client_info JSON from Data_Clients reactives")
    except Exception as exc:
        logger.warning(
            "Could not build client_info from Data_Clients reactives, using empty fallback: %s",
            exc,
        )
        data = {}

    out_path = _REPORTING_DIR / "client_info.json"
    _write_json(out_path, data)
    return out_path


# =========================================================================
# 2.  PORTFOLIO ANALYSIS — inputs & outputs
# =========================================================================


def export_inputs_portfolio_analysis(
    reactives_shiny: dict | None,
) -> Path:
    """Export Portfolio Analysis subtab inputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``inputs_portfolio_analysis.json``.
    """
    # Primary: read from Data_Results["Portfolio_Analysis_Inputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Portfolio_Analysis_Inputs",
    )

    # Fallback: derive from User_Inputs_Shiny when Data_Results not yet populated
    if not data:
        user_inputs = _get_user_inputs(reactives_shiny)
        data = {
            "time_period": _safe_str(user_inputs.get("Portfolio_Analysis_Time_Period", "1Y")),
            "date_range_start": _safe_str(user_inputs.get("Portfolio_Analysis_Date_Range_Start")),
            "date_range_end": _safe_str(user_inputs.get("Portfolio_Analysis_Date_Range_End")),
            "analysis_type": _safe_str(
                user_inputs.get("Portfolio_Analysis_Type", "returns"),
            ),
            "rolling_window": _safe_str(
                user_inputs.get("Portfolio_Analysis_Rolling_Window", "30"),
            ),
            "include_benchmark": bool(
                user_inputs.get("Portfolio_Analysis_Include_Benchmark", True),
            ),
        }

    out_path = _INPUTS_JSON_DIR / "inputs_portfolio_analysis.json"
    _write_json(out_path, data)
    return out_path


def export_outputs_portfolio_analysis(
    reactives_shiny: dict | None,
) -> Path:
    """Export Portfolio Analysis subtab outputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``outputs_portfolio_analysis.json``.
    """
    # Primary: read from Data_Results["Portfolio_Analysis_Outputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Portfolio_Analysis_Outputs",
    )

    # Fallback: derive from Inner_Variables_Shiny when Data_Results not yet populated
    if not data:
        inner = _get_inner_variables(reactives_shiny)
        basic_stats = _safe_reactive_get(inner.get("Portfolio_Analysis_Basic_Stats"))
        quantstats = _safe_reactive_get(inner.get("Portfolio_Analysis_Quantstats_Metrics"))
        basic_list: list[dict[str, Any]] = (
            _polars_to_records(basic_stats)
            if isinstance(basic_stats, pl.DataFrame)
            else (basic_stats if isinstance(basic_stats, list) else [])
        )
        quant_list: list[dict[str, Any]] = (
            _polars_to_records(quantstats)
            if isinstance(quantstats, pl.DataFrame)
            else (quantstats if isinstance(quantstats, list) else [])
        )

        # When no live stats are available, compute from sample CSV portfolio data
        if not basic_list and not quant_list:
            _pv, _bv = _load_sample_portfolio_data()
            if _pv is not None and _pv.height >= 2:
                logger.info("Computing portfolio analysis stats from sample CSV data")
                _pa_stats = _compute_portfolio_analysis_stats_from_values(_pv, _bv)
                basic_list = _pa_stats["basic_statistics"]
                quant_list = _pa_stats["performance_metrics"]

        data = {"basic_statistics": basic_list, "performance_metrics": quant_list}

    out_path = _OUTPUTS_JSON_DIR / "outputs_portfolio_analysis.json"
    _write_json(out_path, data)
    return out_path


# =========================================================================
# 3.  PORTFOLIO COMPARISON — inputs & outputs
# =========================================================================


def export_inputs_portfolio_comparison(
    reactives_shiny: dict | None,
) -> Path:
    """Export Portfolio Comparison subtab inputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``inputs_portfolio_comparison.json``.
    """
    # Primary: read from Data_Results["Portfolio_Comparison_Inputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Portfolio_Comparison_Inputs",
    )

    # Fallback: derive from User_Inputs_Shiny when Data_Results not yet populated
    if not data:
        user_inputs = _get_user_inputs(reactives_shiny)
        data = {
            "time_period": _safe_str(
                user_inputs.get("Portfolio_Comparison_Time_Period", "1Y"),
            ),
            "date_range_start": _safe_str(
                user_inputs.get("Portfolio_Comparison_Date_Range_Start"),
            ),
            "date_range_end": _safe_str(
                user_inputs.get("Portfolio_Comparison_Date_Range_End"),
            ),
            "viz_type": _safe_str(
                user_inputs.get("Portfolio_Comparison_Viz_Type", "normalized"),
            ),
            "show_diff": bool(
                user_inputs.get("Portfolio_Comparison_Show_Diff", False),
            ),
        }

    out_path = _INPUTS_JSON_DIR / "inputs_portfolio_comparison.json"
    _write_json(out_path, data)
    return out_path


def export_outputs_portfolio_comparison(
    reactives_shiny: dict | None,
) -> Path:
    """Export Portfolio Comparison subtab outputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``outputs_portfolio_comparison.json``.
    """
    # Primary: read from Data_Results["Portfolio_Comparison_Outputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Portfolio_Comparison_Outputs",
    )

    # Fallback: derive from Inner_Variables_Shiny when Data_Results not yet populated
    if not data:
        inner = _get_inner_variables(reactives_shiny)
        stats_raw = _safe_reactive_get(inner.get("Portfolio_Comparison_Stats"))
        stats: dict[str, Any] = stats_raw if isinstance(stats_raw, dict) else {}

        # When no live stats are available, compute from sample CSV portfolio data
        # so the report shows illustrative (non-zero) metrics.
        if not stats:
            _pv, _bv = _load_sample_portfolio_data()
            if _pv is not None and _pv.height >= 2:
                logger.info("Computing portfolio comparison metrics from sample CSV data")
                stats = _compute_portfolio_metrics_from_values(_pv, _bv)

        data = {
            "time_period": _safe_str(stats.get("time_period")),
            "start_date": _safe_str(stats.get("start_date")),
            "end_date": _safe_str(stats.get("end_date")),
            "viz_type": _safe_str(stats.get("viz_type", "normalized")),
            "metrics": {
                "total_return": {
                    "portfolio": _safe_float(stats.get("total_return_portfolio")),
                    "benchmark": _safe_float(stats.get("total_return_benchmark")),
                    "difference": _safe_float(stats.get("total_return_difference")),
                },
                "annualized_return": {
                    "portfolio": _safe_float(stats.get("annualized_return_portfolio")),
                    "benchmark": _safe_float(stats.get("annualized_return_benchmark")),
                    "difference": _safe_float(stats.get("annualized_return_difference")),
                },
                "volatility": {
                    "portfolio": _safe_float(stats.get("volatility_portfolio")),
                    "benchmark": _safe_float(stats.get("volatility_benchmark")),
                    "difference": _safe_float(stats.get("volatility_difference")),
                },
                "max_drawdown": {
                    "portfolio": _safe_float(stats.get("max_drawdown_portfolio")),
                    "benchmark": _safe_float(stats.get("max_drawdown_benchmark")),
                    "difference": _safe_float(stats.get("max_drawdown_difference")),
                },
                "sharpe_ratio": {
                    "portfolio": _safe_float(stats.get("sharpe_ratio_portfolio")),
                    "benchmark": _safe_float(stats.get("sharpe_ratio_benchmark")),
                    "difference": _safe_float(stats.get("sharpe_ratio_difference")),
                },
                "correlation": _safe_float(stats.get("correlation")),
                "tracking_error": _safe_float(stats.get("tracking_error")),
                "information_ratio": _safe_float(stats.get("information_ratio")),
            },
        }

    out_path = _OUTPUTS_JSON_DIR / "outputs_portfolio_comparison.json"
    _write_json(out_path, data)
    return out_path


# =========================================================================
# 4.  WEIGHTS ANALYSIS — inputs & outputs
# =========================================================================


def export_inputs_weights_analysis(
    reactives_shiny: dict | None,
) -> Path:
    """Export Weights Distribution subtab inputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``inputs_weights_analysis.json``.
    """
    # Primary: read from Data_Results["Weights_Analysis_Inputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Weights_Analysis_Inputs",
    )

    # Fallback: derive from User_Inputs_Shiny when Data_Results not yet populated
    if not data:
        user_inputs = _get_user_inputs(reactives_shiny)
        selected = _safe_reactive_get(
            user_inputs.get("Weights_Selected_Components"),
        )
        if not isinstance(selected, list):
            selected = []
        data = {
            "time_period": _safe_str(
                user_inputs.get("Weights_Time_Period", "5Y"),
            ),
            "date_range_start": _safe_str(
                user_inputs.get("Weights_Date_Range_Start"),
            ),
            "date_range_end": _safe_str(
                user_inputs.get("Weights_Date_Range_End"),
            ),
            "selected_components": [str(c) for c in selected],
            "viz_type": _safe_str(
                user_inputs.get("Weights_Viz_Type", "area"),
            ),
            "show_pct": bool(user_inputs.get("Weights_Show_Pct", True)),
            "sort_components": bool(
                user_inputs.get("Weights_Sort_Components", True),
            ),
        }

    out_path = _INPUTS_JSON_DIR / "inputs_weights_analysis.json"
    _write_json(out_path, data)
    return out_path


def export_outputs_weights_analysis(
    reactives_shiny: dict | None,
) -> Path:
    """Export Weights Distribution subtab outputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``outputs_weights_analysis.json``.
    """
    # Primary: read from Data_Results["Weights_Analysis_Outputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Weights_Analysis_Outputs",
    )

    # Fallback: derive from Inner_Variables_Shiny when Data_Results not yet populated
    if not data:
        inner = _get_inner_variables(reactives_shiny)
        summary_raw = _safe_reactive_get(inner.get("Weights_Summary_Stats"))
        summary: list[dict[str, Any]] = (
            _polars_to_records(summary_raw)
            if isinstance(summary_raw, pl.DataFrame)
            else (summary_raw if isinstance(summary_raw, list) else [])
        )
        data = {
            "weight_statistics": summary,
        }

    out_path = _OUTPUTS_JSON_DIR / "outputs_weights_analysis.json"
    _write_json(out_path, data)
    return out_path


# =========================================================================
# 5.  SKFOLIO OPTIMIZATION — inputs & outputs
# =========================================================================


def export_inputs_skfolio_optimization(
    reactives_shiny: dict | None,
) -> Path:
    """Export skfolio Optimization subtab inputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``inputs_skfolio_optimization.json``.
    """
    # Primary: read from Data_Results["Portfolio_Optimization_Skfolio_Inputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Portfolio_Optimization_Skfolio_Inputs",
    )

    # Fallback: derive from User_Inputs_Shiny when Data_Results not yet populated
    if not data:
        user_inputs = _get_user_inputs(reactives_shiny)
        data = {
            "time_period": _safe_str(
                user_inputs.get("Skfolio_Time_Period", "3Y"),
            ),
            "method1": {
                "category": _safe_str(
                    user_inputs.get("Skfolio_Method1_Category", "basic"),
                ),
                "type": _safe_str(
                    user_inputs.get("Skfolio_Method1_Type", "equal_weighted"),
                ),
                "objective": _safe_str(
                    user_inputs.get("Skfolio_Method1_Objective"),
                ),
                "risk_aversion": _safe_float(
                    user_inputs.get("Skfolio_Method1_Risk_Aversion"),
                    1.0,
                ),
            },
            "method2": {
                "category": _safe_str(
                    user_inputs.get("Skfolio_Method2_Category", "convex"),
                ),
                "type": _safe_str(
                    user_inputs.get("Skfolio_Method2_Type", "mean_risk"),
                ),
                "objective": _safe_str(
                    user_inputs.get("Skfolio_Method2_Objective"),
                ),
                "risk_aversion": _safe_float(
                    user_inputs.get("Skfolio_Method2_Risk_Aversion"),
                    1.0,
                ),
            },
        }

    out_path = _INPUTS_JSON_DIR / "inputs_skfolio_optimization.json"
    _write_json(out_path, data)
    return out_path


def export_outputs_skfolio_optimization(
    reactives_shiny: dict | None,
) -> Path:
    """Export skfolio Optimization subtab outputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``outputs_skfolio_optimization.json``.
    """
    # Primary: read from Data_Results["Portfolio_Optimization_Skfolio_Outputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Portfolio_Optimization_Skfolio_Outputs",
    )

    # Fallback: derive from Inner_Variables_Shiny when Data_Results not yet populated
    if not data:
        inner = _get_inner_variables(reactives_shiny)
        weights_raw = _safe_reactive_get(inner.get("Skfolio_Results_Table"))
        weights: list[dict[str, Any]] = (
            _polars_to_records(weights_raw)
            if isinstance(weights_raw, pl.DataFrame)
            else (weights_raw if isinstance(weights_raw, list) else [])
        )
        perf_raw = _safe_reactive_get(inner.get("Skfolio_Performance_Summary"))
        perf: dict[str, Any] = perf_raw if isinstance(perf_raw, dict) else {}
        method1_raw = perf.get("method1", {}) if perf else {}
        method2_raw = perf.get("method2", {}) if perf else {}
        performance_summary: dict[str, Any] = {
            "method1": {
                "label": _safe_str(method1_raw.get("label", "Method 1")),
                "annualized_return": _safe_float(method1_raw.get("annualized_return")),
                "volatility": _safe_float(method1_raw.get("volatility")),
                "sharpe_ratio": _safe_float(method1_raw.get("sharpe_ratio")),
            },
            "method2": {
                "label": _safe_str(method2_raw.get("label", "Method 2")),
                "annualized_return": _safe_float(method2_raw.get("annualized_return")),
                "volatility": _safe_float(method2_raw.get("volatility")),
                "sharpe_ratio": _safe_float(method2_raw.get("sharpe_ratio")),
            },
        }
        data = {
            "weights_comparison": weights,
            "statistics_comparison": [],
            "performance_summary": performance_summary,
        }

    out_path = _OUTPUTS_JSON_DIR / "outputs_skfolio_optimization.json"
    _write_json(out_path, data)
    return out_path


# =========================================================================
# 6.  SIMULATION — inputs & outputs
# =========================================================================


def export_inputs_simulation(
    reactives_shiny: dict | None,
) -> Path:
    """Export Simulation subtab inputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``inputs_simulation.json``.
    """
    # Primary: read from Data_Results["Portfolio_Simulation_Inputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Portfolio_Simulation_Inputs",
    )

    # Fallback: derive from User_Inputs_Shiny when Data_Results not yet populated
    if not data:
        user_inputs = _get_user_inputs(reactives_shiny)
        selected = _safe_reactive_get(
            user_inputs.get("Simulation_Selected_Components"),
        )
        if not isinstance(selected, list):
            selected = []
        data = {
            "selected_components": [str(c) for c in selected],
            "num_scenarios": int(
                _safe_float(user_inputs.get("Simulation_Num_Scenarios"), 1000.0),
            ),
            "num_days": int(
                _safe_float(user_inputs.get("Simulation_Num_Days"), 252.0),
            ),
            "start_date": _safe_str(user_inputs.get("Simulation_Start_Date")),
            "initial_value": _safe_float(
                user_inputs.get("Simulation_Initial_Value"),
                100.0,
            ),
            "distribution_type": _safe_str(
                user_inputs.get("Simulation_Distribution_Type", "normal"),
            ),
            "degrees_of_freedom": _safe_float(
                user_inputs.get("Simulation_Degrees_Of_Freedom"),
                5.0,
            ),
            "rng_type": _safe_str(
                user_inputs.get("Simulation_RNG_Type", "pcg64"),
            ),
            "seed": int(_safe_float(user_inputs.get("Simulation_Seed"), 42.0)),
        }

    out_path = _INPUTS_JSON_DIR / "inputs_simulation.json"
    _write_json(out_path, data)
    return out_path


def export_outputs_simulation(
    reactives_shiny: dict | None,
) -> Path:
    """Export Simulation subtab outputs to JSON.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path
        Absolute path to ``outputs_simulation.json``.
    """
    # Primary: read from Data_Results["Portfolio_Simulation_Outputs"]
    data: dict[str, Any] = _get_data_results_value(
        reactives_shiny,
        "Portfolio_Simulation_Outputs",
    )

    # Fallback: derive from Inner_Variables_Shiny when Data_Results not yet populated
    if not data:
        inner = _get_inner_variables(reactives_shiny)
        stats_raw = _safe_reactive_get(inner.get("Simulation_Stats"))
        stats: dict[str, Any] = stats_raw if isinstance(stats_raw, dict) else {}

        # When a simulation has not yet been run, Simulation_Stats is empty and
        # num_scenarios / horizon_days default to 0.  Fall back to the user-input
        # values so that the report body text reads correctly (e.g.
        # "1000 scenarios over 252 trading days") rather than "0 over 0".
        user_inputs = _get_user_inputs(reactives_shiny)
        _fallback_num_scenarios = int(
            _safe_float(
                _safe_reactive_get(
                    user_inputs.get("Input_Tab_Results_Subtab_Simulation_Num_Scenarios"),
                ),
                1000.0,
            ),
        )
        _fallback_horizon_days = int(
            _safe_float(
                _safe_reactive_get(
                    user_inputs.get("Input_Tab_Results_Subtab_Simulation_Num_Days"),
                ),
                252.0,
            ),
        )

        # When no simulation has been run, compute from sample portfolio returns
        if not stats:
            _pv_sim, _ = _load_sample_portfolio_data()
            if _pv_sim is not None and _pv_sim.height >= 2:
                logger.info("Computing simulation stats from sample CSV data")
                stats = _compute_simulation_stats_from_values(
                    _pv_sim,
                    num_scenarios=_fallback_num_scenarios,
                    num_days=_fallback_horizon_days,
                    initial_value=100.0,
                )

        data = {
            "summary_statistics": {
                "num_scenarios": int(_safe_float(stats.get("num_scenarios")))
                or _fallback_num_scenarios,
                "horizon_days": int(_safe_float(stats.get("horizon_days")))
                or _fallback_horizon_days,
                "initial_value": _safe_float(stats.get("initial_value"), 100.0),
                "mean_terminal_value": _safe_float(stats.get("mean_terminal_value")),
                "median_terminal_value": _safe_float(stats.get("median_terminal_value")),
                "std_dev_terminal_value": _safe_float(stats.get("std_dev_terminal_value")),
                "percentile_5": _safe_float(stats.get("percentile_5")),
                "percentile_25": _safe_float(stats.get("percentile_25")),
                "percentile_75": _safe_float(stats.get("percentile_75")),
                "percentile_95": _safe_float(stats.get("percentile_95")),
                "min_terminal_value": _safe_float(stats.get("min_terminal_value")),
                "max_terminal_value": _safe_float(stats.get("max_terminal_value")),
                "probability_of_loss": _safe_float(stats.get("probability_of_loss")),
            },
        }

    out_path = _OUTPUTS_JSON_DIR / "outputs_simulation.json"
    _write_json(out_path, data)
    return out_path


# =========================================================================
# AGGREGATE EXPORT
# =========================================================================


def export_all_report_data(
    reactives_shiny: dict | None,
) -> dict[str, Path]:
    """Export every JSON file required by the Typst report template.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    dict[str, Path]
        Mapping of descriptive name to the absolute path of each written file.
    """
    logger.info("Exporting all report data to JSON files")

    paths: dict[str, Path] = {}
    paths["report_metadata"] = export_report_metadata(reactives_shiny)
    paths["client_info"] = export_client_info(reactives_shiny)
    paths["inputs_portfolio_analysis"] = export_inputs_portfolio_analysis(reactives_shiny)
    paths["outputs_portfolio_analysis"] = export_outputs_portfolio_analysis(reactives_shiny)
    paths["inputs_portfolio_comparison"] = export_inputs_portfolio_comparison(
        reactives_shiny,
    )
    paths["outputs_portfolio_comparison"] = export_outputs_portfolio_comparison(
        reactives_shiny,
    )
    paths["inputs_weights_analysis"] = export_inputs_weights_analysis(reactives_shiny)
    paths["outputs_weights_analysis"] = export_outputs_weights_analysis(reactives_shiny)
    paths["inputs_skfolio_optimization"] = export_inputs_skfolio_optimization(
        reactives_shiny,
    )
    paths["outputs_skfolio_optimization"] = export_outputs_skfolio_optimization(
        reactives_shiny,
    )
    paths["inputs_simulation"] = export_inputs_simulation(reactives_shiny)
    paths["outputs_simulation"] = export_outputs_simulation(reactives_shiny)

    logger.info("Exported %d JSON files for report generation", len(paths))
    return paths


# =========================================================================
# helpers to access reactives
# =========================================================================


def _get_user_inputs(reactives_shiny: dict | None) -> dict[str, Any]:
    """Safely extract ``User_Inputs_Shiny`` from reactives dict."""
    if not reactives_shiny or not isinstance(reactives_shiny, dict):
        return {}
    raw = reactives_shiny.get("User_Inputs_Shiny", {})
    return raw if isinstance(raw, dict) else {}


def _get_inner_variables(reactives_shiny: dict | None) -> dict[str, Any]:
    """Safely extract ``Inner_Variables_Shiny`` from reactives dict."""
    if not reactives_shiny or not isinstance(reactives_shiny, dict):
        return {}
    raw = reactives_shiny.get("Inner_Variables_Shiny", {})
    return raw if isinstance(raw, dict) else {}


def _get_data_results_value(
    reactives_shiny: dict | None,
    subtab_key: str,
) -> dict[str, Any]:
    """Read one ``Data_Results`` subtab key as a JSON-serialisable dict.

    Calls :func:`~src.dashboard.shiny_utils.utils_reporting.build_results_data_json_from_reactives`
    and returns a JSON-serialisable dict.  Returns an empty dict on any error
    (including when the subtab value has not yet been populated).

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.
    subtab_key : str
        One of the 10 valid keys in ``Data_Results``
        (e.g. ``"Portfolio_Analysis_Inputs"``).

    Returns
    -------
    dict[str, Any]
        JSON-serialisable dict for the subtab key, or ``{}`` when not yet
        populated or on error.
    """
    if not reactives_shiny or not isinstance(reactives_shiny, dict):
        return {}
    try:
        from src.dashboard.shiny_utils.utils_reporting import (
            build_results_data_json_from_reactives,
        )

        return build_results_data_json_from_reactives(reactives_shiny, subtab_key)
    except Exception as exc:
        logger.debug(
            "_get_data_results_value: could not get '%s' from Data_Results: %s",
            subtab_key,
            exc,
        )
        return {}
