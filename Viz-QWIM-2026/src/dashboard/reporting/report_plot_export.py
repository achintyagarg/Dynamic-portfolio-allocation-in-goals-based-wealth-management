"""Static Plot Export Module for QWIM Report Generation.

Recreates dashboard Plotly charts as static SVG images using ``plotnine``
(a Python implementation of ggplot2).  The SVG files are saved to
``src/dashboard/reporting/outputs_images/`` and referenced by the Typst
template via ``#image(...)`` calls.

Each public function:
1. Extracts the relevant data from ``reactives_shiny``.
2. Builds a ``plotnine.ggplot`` object.
3. Saves it as an SVG at the canonical path.

Dependencies
------------
* plotnine: ggplot2-style plotting
* polars: DataFrame handling
* numpy: Numeric utilities
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from plotnine import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_area,
    geom_col,
    geom_histogram,
    geom_hline,
    geom_line,
    geom_ribbon,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_date,
    theme,
    theme_minimal,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPORTING_DIR = Path(__file__).resolve().parent
_IMAGES_DIR = _REPORTING_DIR / "outputs_images"

# Sample data paths (used when live dashboard data is unavailable)
_PROJECT_ROOT_PLOTS = Path(__file__).resolve().parents[3]
_PROCESSED_DIR_PLOTS = _PROJECT_ROOT_PLOTS / "inputs" / "processed"
_SAMPLE_PORTFOLIO_CSV_PLOTS = _PROCESSED_DIR_PLOTS / "sample_portfolio_values.csv"
_BENCHMARK_PORTFOLIO_CSV_PLOTS = _PROCESSED_DIR_PLOTS / "benchmark_portfolio_values.csv"

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

_QWIM_THEME = theme_minimal() + theme(
    plot_title=element_text(size=14, weight="bold"),
    axis_title=element_text(size=11),
    axis_text=element_text(size=9),
    legend_title=element_text(size=10, weight="bold"),
    legend_text=element_text(size=9),
    panel_grid_minor=element_blank(),
    panel_grid_major=element_line(color="#e0e0e0", size=0.4),
    plot_background=element_rect(fill="white", color="white"),
    figure_size=(10, 6),
)

_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # teal
    "#aec7e8",  # light-blue
    "#ffbb78",  # light-orange
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Canonical SVG filenames expected by the Typst template
_EXPECTED_SVG_FILES: dict[str, str] = {
    "portfolio_analysis": "portfolio_analysis.svg",
    "portfolio_comparison": "portfolio_comparison.svg",
    "weights_analysis": "weights_analysis.svg",
    "weights_composition": "weights_composition.svg",
    "skfolio_weights": "skfolio_weights.svg",
    "skfolio_performance": "skfolio_performance.svg",
    "simulation_fan_chart": "simulation_fan_chart.svg",
    "simulation_histogram": "simulation_histogram.svg",
}


def _ensure_dir() -> None:
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_all_svg_files_exist() -> dict[str, str]:
    """Guarantee every SVG expected by the Typst template exists on disk.

    For any file that is missing or empty (0 bytes), a placeholder SVG
    is created.  Returns a mapping of chart key to status:

    * ``"ok"`` — real chart SVG already present
    * ``"placeholder"`` — placeholder was created (data was unavailable)

    This MUST be called after all chart-export functions and before
    Typst compilation.

    Returns
    -------
    dict[str, str]
        ``{chart_key: "ok"|"placeholder"}``.
    """
    _ensure_dir()
    status: dict[str, str] = {}
    for key, filename in _EXPECTED_SVG_FILES.items():
        svg_path = _IMAGES_DIR / filename
        if svg_path.exists() and svg_path.stat().st_size > 0:
            status[key] = "ok"
        else:
            _create_placeholder_svg(
                filename,
                title=key.replace("_", " ").title(),
            )
            status[key] = "placeholder"
            logger.warning("Created placeholder SVG for missing chart: %s", key)
    return status


def _create_placeholder_svg(filename: str, title: str = "Chart Unavailable") -> Path:
    """Create a minimal placeholder SVG when chart data is unavailable.

    This prevents Typst ``image()`` calls from crashing on missing files.

    Parameters
    ----------
    filename : str
        SVG filename to create inside ``_IMAGES_DIR``.
    title : str
        Text displayed in the placeholder.

    Returns
    -------
    Path
        Absolute path to the created placeholder SVG.
    """
    _ensure_dir()
    svg_path = _IMAGES_DIR / filename
    svg_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400" '
        'viewBox="0 0 800 400">\n'
        '  <rect width="800" height="400" fill="#f8fafc" stroke="#cbd5e1" '
        'stroke-width="2" rx="8"/>\n'
        f'  <text x="400" y="190" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="18" fill="#64748b">{title}</text>\n'
        '  <text x="400" y="220" text-anchor="middle" font-family="Arial, sans-serif" '
        'font-size="12" fill="#94a3b8">Insufficient data to render this chart</text>\n'
        "</svg>\n"
    )
    svg_path.write_text(svg_content, encoding="utf-8")
    logger.info("Created placeholder SVG: %s", filename)
    return svg_path


def _safe_reactive_get(reactive_value: Any) -> Any:
    if reactive_value is None:
        return None
    if hasattr(reactive_value, "get"):
        try:
            return reactive_value.get()
        except Exception:
            return None
    return reactive_value


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _get_inner_variables(reactives_shiny: dict | None) -> dict[str, Any]:
    if not reactives_shiny or not isinstance(reactives_shiny, dict):
        return {}
    raw = reactives_shiny.get("Inner_Variables_Shiny", {})
    return raw if isinstance(raw, dict) else {}


def _get_visual_objects(reactives_shiny: dict | None) -> dict[str, Any]:
    if not reactives_shiny or not isinstance(reactives_shiny, dict):
        return {}
    raw = reactives_shiny.get("Visual_Objects_Shiny", {})
    return raw if isinstance(raw, dict) else {}


# ---------------------------------------------------------------------------
# Sample-data helpers for plot fallback
# ---------------------------------------------------------------------------


def _load_sample_csv_for_plots() -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load sample portfolio and benchmark CSVs for plot fallback."""

    def _load(path: Path) -> pl.DataFrame | None:
        try:
            df = pl.read_csv(path)
            df = df.with_columns(
                pl.col("Date").str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d").alias("Date"),
            )
            df = df.rename({"Value": "portfolio_value"})
            return df.sort("Date")
        except Exception as exc:
            logger.debug("Could not load sample CSV %s: %s", path, exc)
            return None

    return _load(_SAMPLE_PORTFOLIO_CSV_PLOTS), _load(_BENCHMARK_PORTFOLIO_CSV_PLOTS)


def _build_simulation_fan_data(
    pv: pl.DataFrame,
    num_scenarios: int = 1000,
    num_days: int = 252,
    initial_value: float = 100.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Simulate portfolio paths and return per-day percentile bands.

    Returns
    -------
    pl.DataFrame
        Columns: ``day``, ``p5``, ``p25``, ``median``, ``mean``, ``p75``, ``p95``.
    """
    pv_arr = pv["portfolio_value"].to_numpy()
    rets = np.diff(pv_arr) / pv_arr[:-1]
    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1))

    rng = np.random.default_rng(seed)
    sim_rets = rng.normal(mu, sigma, size=(num_scenarios, num_days))
    # Cumulative product along days → shape (num_scenarios, num_days)
    cum_rets = np.cumprod(1.0 + sim_rets, axis=1) * initial_value

    days = np.arange(1, num_days + 1)
    data = {
        "day": days,
        "p5": np.percentile(cum_rets, 5, axis=0),
        "p25": np.percentile(cum_rets, 25, axis=0),
        "median": np.percentile(cum_rets, 50, axis=0),
        "mean": np.mean(cum_rets, axis=0),
        "p75": np.percentile(cum_rets, 75, axis=0),
        "p95": np.percentile(cum_rets, 95, axis=0),
    }
    return pl.DataFrame(data)


def _save_plot(plot: ggplot, filename: str, width: float = 10, height: float = 6) -> Path:
    """Save a plotnine ggplot as SVG and return the path."""
    _ensure_dir()
    out_path = _IMAGES_DIR / filename
    plot.save(
        str(out_path),
        width=width,
        height=height,
        dpi=150,
        verbose=False,
    )
    logger.info("Saved SVG: %s (%d bytes)", filename, out_path.stat().st_size)
    return out_path


# =========================================================================
# 1. PORTFOLIO ANALYSIS — main chart
# =========================================================================


def export_plot_portfolio_analysis(
    reactives_shiny: dict | None,
) -> Path | None:
    """Render the Portfolio Analysis main chart as SVG.

    Attempts to reconstruct a *returns distribution* histogram from stored
    data.  Falls back to a placeholder chart when data is unavailable.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path | None
        Path to SVG file, or ``None`` if data is insufficient.
    """
    inner = _get_inner_variables(reactives_shiny)

    portfolio_values = _safe_reactive_get(inner.get("Portfolio_Values"))
    benchmark_values = _safe_reactive_get(inner.get("Benchmark_Values"))

    if not isinstance(portfolio_values, pl.DataFrame) or portfolio_values.height < 2:
        logger.info("Loading sample CSV for portfolio analysis plot")
        portfolio_values, benchmark_values = _load_sample_csv_for_plots()

    if not isinstance(portfolio_values, pl.DataFrame) or portfolio_values.height < 2:
        logger.warning("Not enough portfolio data for analysis plot")
        return None

    # Compute daily returns
    pv = portfolio_values.sort("Date")
    returns_col = "portfolio_value" if "portfolio_value" in pv.columns else pv.columns[1]
    pv = pv.with_columns(
        (pl.col(returns_col) / pl.col(returns_col).shift(1) - 1.0).alias("return"),
    ).drop_nulls("return")

    pdf = pv.select(["return"]).with_columns(pl.lit("Portfolio").alias("series")).to_pandas()

    if isinstance(benchmark_values, pl.DataFrame) and benchmark_values.height >= 2:
        bv = benchmark_values.sort("Date")
        bv_col = "portfolio_value" if "portfolio_value" in bv.columns else bv.columns[1]
        bv = bv.with_columns(
            (pl.col(bv_col) / pl.col(bv_col).shift(1) - 1.0).alias("return"),
        ).drop_nulls("return")
        bpdf = (
            bv.select(["return"])
            .with_columns(
                pl.lit("Benchmark").alias("series"),
            )
            .to_pandas()
        )
        import pandas as pd  # only for plotnine compat

        pdf = pd.concat([pdf, bpdf], ignore_index=True)

    p = (
        ggplot(pdf, aes(x="return", fill="series"))
        + geom_histogram(alpha=0.6, bins=50, position="identity")
        + scale_fill_manual(values=["#1f77b4", "#ff7f0e"])
        + labs(
            title="Returns Distribution",
            x="Daily Return",
            y="Frequency",
            fill="Series",
        )
        + _QWIM_THEME
    )

    return _save_plot(p, "portfolio_analysis.svg")


# =========================================================================
# 2. PORTFOLIO COMPARISON — main chart
# =========================================================================


def export_plot_portfolio_comparison(
    reactives_shiny: dict | None,
) -> Path | None:
    """Render Portfolio vs Benchmark normalised value chart as SVG.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path | None
        Path to SVG file, or ``None`` if data is insufficient.
    """
    inner = _get_inner_variables(reactives_shiny)

    portfolio_values = _safe_reactive_get(inner.get("Portfolio_Values"))
    benchmark_values = _safe_reactive_get(inner.get("Benchmark_Values"))

    if not isinstance(portfolio_values, pl.DataFrame) or portfolio_values.height < 2:
        logger.info("Loading sample CSV for portfolio comparison plot")
        portfolio_values, benchmark_values = _load_sample_csv_for_plots()

    if not isinstance(portfolio_values, pl.DataFrame) or portfolio_values.height < 2:
        logger.warning("Not enough data for comparison plot")
        return None

    pv = portfolio_values.sort("Date")
    pv_col = "portfolio_value" if "portfolio_value" in pv.columns else pv.columns[1]

    # Normalise to base 100
    start_val = pv[pv_col][0]
    if start_val == 0:
        start_val = 1.0
    pv = pv.with_columns(
        (pl.col(pv_col) / start_val * 100).alias("normalised"),
    )
    pdf_p = (
        pv.select(["Date", "normalised"])
        .with_columns(
            pl.lit("Portfolio").alias("series"),
        )
        .to_pandas()
    )
    pdf_p["Date"] = __import__("pandas").to_datetime(pdf_p["Date"])

    frames = [pdf_p]

    if isinstance(benchmark_values, pl.DataFrame) and benchmark_values.height >= 2:
        bv = benchmark_values.sort("Date")
        bv_col = "portfolio_value" if "portfolio_value" in bv.columns else bv.columns[1]
        bv_start = bv[bv_col][0]
        if bv_start == 0:
            bv_start = 1.0
        bv = bv.with_columns(
            (pl.col(bv_col) / bv_start * 100).alias("normalised"),
        )
        pdf_b = (
            bv.select(["Date", "normalised"])
            .with_columns(
                pl.lit("Benchmark").alias("series"),
            )
            .to_pandas()
        )
        pdf_b["Date"] = __import__("pandas").to_datetime(pdf_b["Date"])
        frames.append(pdf_b)

    import pandas as pd

    combined = pd.concat(frames, ignore_index=True)

    p = (
        ggplot(combined, aes(x="Date", y="normalised", color="series"))
        + geom_line(size=1)
        + scale_color_manual(values=["#ff7f0e", "#1f77b4"])
        + geom_hline(yintercept=100, linetype="dashed", color="#888888", size=0.5)
        + labs(
            title="Portfolio vs Benchmark (Normalised, Base = 100)",
            x="Date",
            y="Value (Base = 100)",
            color="Series",
        )
        + scale_x_date(date_labels="%Y-%m")
        + _QWIM_THEME
    )

    return _save_plot(p, "portfolio_comparison.svg")


# =========================================================================
# 3. Weights Analysis — stacked area + pie
# =========================================================================


def export_plot_weights_analysis(
    reactives_shiny: dict | None,
) -> Path | None:
    """Render the Weights Analysis stacked-area chart as SVG.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path | None
        Path to SVG file, or ``None`` if data is insufficient.
    """
    inner = _get_inner_variables(reactives_shiny)

    weights_df = _safe_reactive_get(inner.get("Weights_Data"))
    if not isinstance(weights_df, pl.DataFrame) or weights_df.height == 0:
        logger.warning("No weights data for distribution plot")
        return None

    # Melt to long form: Date, component, weight
    value_cols = [c for c in weights_df.columns if c != "Date"]
    if not value_cols:
        return None

    long = weights_df.unpivot(
        index="Date",
        on=value_cols,
        variable_name="component",
        value_name="weight",
    ).to_pandas()
    long["Date"] = __import__("pandas").to_datetime(long["Date"])

    num_components = len(value_cols)
    colors = (_PALETTE * ((num_components // len(_PALETTE)) + 1))[:num_components]

    p = (
        ggplot(long, aes(x="Date", y="weight", fill="component"))
        + geom_area(alpha=0.85, position="stack")
        + scale_fill_manual(values=colors)
        + labs(
            title="Portfolio Weights Analysis Over Time",
            x="Date",
            y="Weight",
            fill="Component",
        )
        + scale_x_date(date_labels="%Y-%m")
        + _QWIM_THEME
    )

    return _save_plot(p, "weights_analysis.svg")


def export_plot_weights_pie(
    reactives_shiny: dict | None,
) -> Path | None:
    """Render current portfolio composition as a bar chart (plotnine lacks native pie).

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path | None
        Path to SVG file, or ``None`` if data is insufficient.
    """
    inner = _get_inner_variables(reactives_shiny)

    weights_df = _safe_reactive_get(inner.get("Weights_Data"))
    if not isinstance(weights_df, pl.DataFrame) or weights_df.height == 0:
        return None

    value_cols = [c for c in weights_df.columns if c != "Date"]
    if not value_cols:
        return None

    # Latest row
    last_row = weights_df.sort("Date").tail(1)
    data_dict: dict[str, list[Any]] = {"component": [], "weight": []}
    for col in value_cols:
        data_dict["component"].append(col)
        data_dict["weight"].append(float(last_row[col][0]))

    import pandas as pd

    pdf = pd.DataFrame(data_dict)

    num_components = len(value_cols)
    colors = (_PALETTE * ((num_components // len(_PALETTE)) + 1))[:num_components]

    p = (
        ggplot(pdf, aes(x="component", y="weight", fill="component"))
        + geom_col(alpha=0.85)
        + scale_fill_manual(values=colors)
        + labs(
            title="Current Portfolio Composition",
            x="Component",
            y="Weight",
            fill="Component",
        )
        + _QWIM_THEME
        + theme(figure_size=(10, 5))
    )

    return _save_plot(p, "weights_composition.svg", width=10, height=5)


# =========================================================================
# 4. SKFOLIO OPTIMIZATION — weights bar + performance line
# =========================================================================


def export_plot_skfolio_weights(
    reactives_shiny: dict | None,
) -> Path | None:
    """Render skfolio optimised weights comparison as grouped bar chart.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path | None
        Path to SVG file, or ``None`` if data is insufficient.
    """
    inner = _get_inner_variables(reactives_shiny)

    results_raw = _safe_reactive_get(inner.get("Skfolio_Results_Table"))
    if results_raw is None:
        return None

    import pandas as pd

    if isinstance(results_raw, pl.DataFrame):
        pdf = results_raw.to_pandas()
    elif isinstance(results_raw, pd.DataFrame):
        pdf = results_raw
    elif isinstance(results_raw, list):
        pdf = pd.DataFrame(results_raw)
    else:
        return None

    if pdf.empty:
        return None

    # Expect columns: Asset, Portfolio_1, Portfolio_2 (or similar)
    asset_col = pdf.columns[0]
    if len(pdf.columns) < 3:
        return None

    p1_col = pdf.columns[1]
    p2_col = pdf.columns[2]

    melted = pdf.melt(
        id_vars=[asset_col],
        value_vars=[p1_col, p2_col],
        var_name="portfolio",
        value_name="weight",
    )
    melted = melted.rename(columns={asset_col: "asset"})

    p = (
        ggplot(melted, aes(x="asset", y="weight", fill="portfolio"))
        + geom_col(position="dodge", alpha=0.85)
        + scale_fill_manual(values=["#1f77b4", "#ff7f0e"])
        + labs(
            title="Optimised Portfolio Weights Comparison",
            x="Asset",
            y="Weight (%)",
            fill="Portfolio",
        )
        + _QWIM_THEME
        + theme(axis_text_x=element_text(angle=45, ha="right"))
    )

    return _save_plot(p, "skfolio_weights.svg")


def export_plot_skfolio_performance(
    reactives_shiny: dict | None,
) -> Path | None:
    """Render skfolio normalised performance comparison as line chart.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path | None
        Path to SVG file, or ``None`` if data is insufficient.
    """
    inner = _get_inner_variables(reactives_shiny)

    perf_data = _safe_reactive_get(inner.get("Skfolio_Performance_Data"))
    if not isinstance(perf_data, pl.DataFrame) or perf_data.height < 2:
        return None

    import pandas as pd

    pdf = perf_data.to_pandas()
    pdf["Date"] = pd.to_datetime(pdf["Date"])

    value_cols = [c for c in pdf.columns if c != "Date"]
    if not value_cols:
        return None

    melted = pdf.melt(
        id_vars=["Date"],
        value_vars=value_cols,
        var_name="portfolio",
        value_name="value",
    )

    colors = ["#1f77b4", "#ff7f0e"][: len(value_cols)]

    p = (
        ggplot(melted, aes(x="Date", y="value", color="portfolio"))
        + geom_line(size=1)
        + scale_color_manual(values=colors)
        + geom_hline(yintercept=100, linetype="dashed", color="#888888", size=0.5)
        + labs(
            title="Optimised Portfolio Performance (Normalised, Base = 100)",
            x="Date",
            y="Value (Base = 100)",
            color="Portfolio",
        )
        + scale_x_date(date_labels="%Y-%m")
        + _QWIM_THEME
    )

    return _save_plot(p, "skfolio_performance.svg")


# =========================================================================
# 5. SIMULATION — fan chart + histogram
# =========================================================================


def export_plot_simulation_fan_chart(
    reactives_shiny: dict | None,
) -> Path | None:
    """Render the Monte Carlo simulation fan chart as SVG.

    The fan chart shows P5-P95 and P25-P75 confidence bands plus median.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path | None
        Path to SVG file, or ``None`` if data is insufficient.
    """
    inner = _get_inner_variables(reactives_shiny)

    fan_data = _safe_reactive_get(inner.get("Simulation_Fan_Data"))
    if not isinstance(fan_data, pl.DataFrame) or fan_data.height < 2:
        logger.info("Building simulation fan data from sample CSV")
        _pv_fan, _ = _load_sample_csv_for_plots()
        if _pv_fan is not None and _pv_fan.height >= 2:
            fan_data = _build_simulation_fan_data(_pv_fan)

    if not isinstance(fan_data, pl.DataFrame) or fan_data.height < 2:
        logger.warning("No simulation fan data for fan chart")
        return None

    pdf = fan_data.to_pandas()
    if "day" not in pdf.columns:
        pdf["day"] = range(len(pdf))

    p = ggplot(pdf, aes(x="day"))

    # Outer band: P5-P95
    if "p5" in pdf.columns and "p95" in pdf.columns:
        p = p + geom_ribbon(aes(ymin="p5", ymax="p95"), fill="#1f77b4", alpha=0.15)

    # Inner band: P25-P75
    if "p25" in pdf.columns and "p75" in pdf.columns:
        p = p + geom_ribbon(aes(ymin="p25", ymax="p75"), fill="#1f77b4", alpha=0.3)

    # Median line
    if "median" in pdf.columns:
        p = p + geom_line(aes(y="median"), color="#1f77b4", size=1.2)

    # Mean line (dashed)
    if "mean" in pdf.columns:
        p = p + geom_line(aes(y="mean"), color="#ff7f0e", size=1, linetype="dashed")

    # Initial value reference
    initial = _safe_float(pdf.get("median", [100])[0] if "median" in pdf.columns else 100)
    p = p + geom_hline(yintercept=initial, linetype="dotted", color="#888888", size=0.5)

    p = (
        p
        + labs(
            title="Monte Carlo Simulation — Portfolio Value Paths",
            x="Trading Day",
            y="Portfolio Value ($)",
        )
        + _QWIM_THEME
    )

    return _save_plot(p, "simulation_fan_chart.svg")


def export_plot_simulation_histogram(
    reactives_shiny: dict | None,
) -> Path | None:
    """Render the terminal-value distribution histogram as SVG.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    Path | None
        Path to SVG file, or ``None`` if data is insufficient.
    """
    inner = _get_inner_variables(reactives_shiny)

    terminal_values = _safe_reactive_get(inner.get("Simulation_Terminal_Values"))
    if terminal_values is None:
        logger.info("Building simulation terminal values from sample CSV")
        _pv_hist, _ = _load_sample_csv_for_plots()
        if _pv_hist is not None and _pv_hist.height >= 2:
            pv_arr = _pv_hist["portfolio_value"].to_numpy()
            rets = np.diff(pv_arr) / pv_arr[:-1]
            mu = float(np.mean(rets))
            sigma = float(np.std(rets, ddof=1))
            rng = np.random.default_rng(42)
            terminal_values = 100.0 * np.prod(
                1.0 + rng.normal(mu, sigma, size=(1000, 252)),
                axis=1,
            )

    if terminal_values is None:
        return None

    import pandas as pd

    if isinstance(terminal_values, pl.DataFrame):
        vals = terminal_values.to_pandas()
    elif isinstance(terminal_values, (list, np.ndarray)):
        vals = pd.DataFrame({"terminal_value": terminal_values})
    else:
        return None

    if vals.empty:
        return None

    val_col = "terminal_value" if "terminal_value" in vals.columns else vals.columns[0]
    vals = vals.rename(columns={val_col: "terminal_value"})

    mean_val = float(vals["terminal_value"].mean())
    median_val = float(vals["terminal_value"].median())

    p = (
        ggplot(vals, aes(x="terminal_value"))
        + geom_histogram(bins=50, fill="#1f77b4", alpha=0.7, color="white")
        + geom_vline(xintercept=mean_val, linetype="dashed", color="#ff7f0e", size=1)
        + geom_vline(xintercept=median_val, linetype="solid", color="#2ca02c", size=1)
        + labs(
            title="Terminal Value Distribution",
            x="Terminal Portfolio Value ($)",
            y="Frequency",
        )
        + _QWIM_THEME
        + theme(figure_size=(10, 5))
    )

    return _save_plot(p, "simulation_histogram.svg", width=10, height=5)


# =========================================================================
# AGGREGATE EXPORT
# =========================================================================


def export_all_report_plots(
    reactives_shiny: dict | None,
) -> dict[str, Path | None]:
    """Generate every SVG image required by the Typst report template.

    Parameters
    ----------
    reactives_shiny : dict | None
        Shared reactive state dictionary.

    Returns
    -------
    dict[str, Path | None]
        Mapping of descriptive name to path (``None`` where data is missing).
    """
    logger.info("Exporting all report plots to SVG")

    paths: dict[str, Path | None] = {}
    paths["portfolio_analysis"] = export_plot_portfolio_analysis(reactives_shiny)
    paths["portfolio_comparison"] = export_plot_portfolio_comparison(reactives_shiny)
    paths["weights_analysis"] = export_plot_weights_analysis(reactives_shiny)
    paths["weights_composition"] = export_plot_weights_pie(reactives_shiny)
    paths["skfolio_weights"] = export_plot_skfolio_weights(reactives_shiny)
    paths["skfolio_performance"] = export_plot_skfolio_performance(reactives_shiny)
    paths["simulation_fan_chart"] = export_plot_simulation_fan_chart(reactives_shiny)
    paths["simulation_histogram"] = export_plot_simulation_histogram(reactives_shiny)

    generated = sum(1 for v in paths.values() if v is not None)
    logger.info("Generated %d / %d SVG images", generated, len(paths))

    # Create placeholder SVGs for any charts that could not be generated so
    # that the Typst template's image() calls never fail on missing files.
    _placeholder_map: dict[str, str] = {
        "portfolio_analysis": "portfolio_analysis.svg",
        "portfolio_comparison": "portfolio_comparison.svg",
        "weights_analysis": "weights_analysis.svg",
        "weights_composition": "weights_composition.svg",
        "skfolio_weights": "skfolio_weights.svg",
        "skfolio_performance": "skfolio_performance.svg",
        "simulation_fan_chart": "simulation_fan_chart.svg",
        "simulation_histogram": "simulation_histogram.svg",
    }
    for key, svg_filename in _placeholder_map.items():
        if paths.get(key) is None:
            paths[key] = _create_placeholder_svg(
                svg_filename,
                title=key.replace("_", " ").title(),
            )

    return paths
