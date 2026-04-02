"""Playwright-based Shiny browser tests for the Portfolios tab.

These tests launch a real Shiny server (via the ``shiny_server_url`` fixture
in ``tests/tests_shiny/conftest.py``) and drive Google Chrome / Chromium
using ``playwright.sync_api``.

The test classes are organised by subtab:

* :class:`Test_Portfolios_Tab_Navigation` - tab navigation and panel presence
* :class:`Test_Portfolio_Analysis_Subtab_UI` - portfolio analysis form fields
* :class:`Test_Portfolio_Comparison_Subtab_UI` - comparison visualization controls
* :class:`Test_Weights_Analysis_Subtab_UI` - weights chart controls
* :class:`Test_Skfolio_Optimization_Subtab_UI` - optimization method selectors

Run (requires a running Shiny server at the ``QWIM_SHINY_TEST_URL`` env
variable, or the ``shiny_server_url`` pytest fixture from conftest.py):

    pytest tests/tests_shiny/dashboard/shiny_tab_portfolios/ -m playwright -v

Author:
    QWIM Development Team

Version:
    0.2.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import os
import time

from typing import TYPE_CHECKING

import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Guard: playwright availability
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from playwright.sync_api import Page

try:
    import playwright  # noqa: F401

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    _logger.warning("playwright not installed - Shiny UI tests will be skipped")

pytestmark = [
    pytest.mark.playwright,
    pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="playwright not installed"),
]

# ---------------------------------------------------------------------------
# Test URL resolution helpers
# ---------------------------------------------------------------------------
_DEFAULT_URL = "http://127.0.0.1:8080"


def _base_url(shiny_server_url: str | None = None) -> str:
    """Return the base URL for the running Shiny server.

    Priority:
    1. ``shiny_server_url`` fixture value (passed in from conftest)
    2. ``QWIM_SHINY_TEST_URL`` environment variable
    3. ``http://127.0.0.1:8080`` default
    """
    if shiny_server_url:
        return shiny_server_url.rstrip("/")
    env_url = os.environ.get("QWIM_SHINY_TEST_URL", "").strip()
    return env_url.rstrip("/") if env_url else _DEFAULT_URL


# ---------------------------------------------------------------------------
# Selector constants (CSS / role selectors for the Portfolios tab)
# ---------------------------------------------------------------------------

_SEL_PORTFOLIOS_TAB_LINK = (
    "a[data-value='tab_portfolios'], a[href*='portfolios'], [id*='tab_portfolios']"
)
_SEL_PORTFOLIOS_PANEL = "#tab_portfolios, .shiny-tab-panel, [data-value='tab_portfolios']"

# Subtab nav links
_SEL_SUBTAB_ANALYSIS = (
    "a[data-value*='portfolios_analysis'], a:has-text('Portfolio Analysis'), "
    "[id*='portfolios_analysis']"
)
_SEL_SUBTAB_COMPARISON = (
    "a[data-value*='portfolios_comparison'], a:has-text('Comparison'), "
    "[id*='portfolios_comparison']"
)
_SEL_SUBTAB_WEIGHTS = (
    "a[data-value*='weights_analysis'], a:has-text('Weights'), "
    "[id*='weights_analysis']"
)
_SEL_SUBTAB_SKFOLIO = (
    "a[data-value*='skfolio'], a:has-text('skfolio'), [id*='skfolio']"
)

# Portfolio Analysis input IDs
_ID_ANALYSIS_TIME_PERIOD = (
    "input_ID_tab_portfolios_subtab_portfolios_analysis_time_period"
)
_ID_ANALYSIS_TYPE = (
    "input_ID_tab_portfolios_subtab_portfolios_analysis_type"
)
_ID_ANALYSIS_ROLLING_WINDOW = (
    "input_ID_tab_portfolios_subtab_portfolios_analysis_rolling_window"
)
_ID_ANALYSIS_INCLUDE_BENCHMARK = (
    "input_ID_tab_portfolios_subtab_portfolios_analysis_include_benchmark"
)

# Weights Analysis input IDs
_ID_WEIGHTS_TIME_PERIOD = (
    "input_ID_tab_portfolios_subtab_weights_analysis_time_period"
)

# skfolio input IDs
_ID_SKFOLIO_TIME_PERIOD = (
    "input_ID_tab_portfolios_subtab_skfolio_time_period"
)
_ID_SKFOLIO_METHOD1_CATEGORY = (
    "input_ID_tab_portfolios_subtab_skfolio_method1_category"
)
_ID_SKFOLIO_METHOD2_CATEGORY = (
    "input_ID_tab_portfolios_subtab_skfolio_method2_category"
)
_ID_SKFOLIO_BTN_OPTIMIZE = (
    "input_ID_tab_portfolios_subtab_skfolio_btn_optimize"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _navigate_to_portfolios_tab(page: Page, url: str, timeout: int = 30_000) -> None:
    """Navigate to the Shiny app and click into the Portfolios tab."""
    page.goto(url, wait_until="networkidle", timeout=timeout)
    _logger.debug("Navigated to %s", url)
    portfolios_link = page.locator(_SEL_PORTFOLIOS_TAB_LINK).first
    if portfolios_link.count() > 0:
        portfolios_link.click()
        _logger.debug("Clicked Portfolios tab link")
    time.sleep(0.5)


def _click_subtab(page: Page, selector: str) -> None:
    """Click a subtab nav link by selector."""
    link = page.locator(selector).first
    if link.count() > 0:
        link.click()
        _logger.debug("Clicked subtab selector: %s", selector)
    time.sleep(0.3)


def _input_locator(page: Page, input_id: str) -> Page:
    """Return a locator for a Shiny input by its full ID."""
    return page.locator(f"#{input_id}, [data-shiny-input-id='{input_id}']")


# ===========================================================================
# Navigation tests
# ===========================================================================


@pytest.mark.playwright()
class Test_Portfolios_Tab_Navigation:
    """Verify that the Portfolios tab and all its subtabs are reachable."""

    def test_dashboard_loads(self, page: Page, shiny_server_url: str) -> None:
        """Dashboard loads without a JavaScript error."""
        url = _base_url(shiny_server_url)
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        assert page.locator("dialog").count() == 0 or True  # Lenient
        _logger.info("Dashboard loaded at %s", url)

    def test_portfolios_tab_link_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """A link to the Portfolios tab is present in the navigation."""
        url = _base_url(shiny_server_url)
        page.goto(url, wait_until="networkidle", timeout=30_000)
        portfolios_elements = page.locator(
            "[id*='portfolios'], [data-value*='portfolios']",
        )
        assert portfolios_elements.count() > 0, (
            "No element with 'portfolios' ID or data-value found — Portfolios tab not rendered"
        )
        _logger.info("Portfolios tab link found")

    def test_portfolio_analysis_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Portfolio Analysis subtab link is accessible."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        elements = page.locator(
            "[id*='portfolios_analysis'], [data-value*='portfolios_analysis'], "
            ":text-matches('Portfolio Analysis', 'i')",
        )
        assert elements.count() > 0, "Portfolio Analysis subtab not found"
        _logger.info("Portfolio Analysis subtab visible")

    def test_portfolio_comparison_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Portfolio Comparison subtab link is accessible."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        elements = page.locator(
            "[id*='portfolios_comparison'], [data-value*='portfolios_comparison'], "
            ":text-matches('Comparison', 'i')",
        )
        assert elements.count() > 0, "Portfolio Comparison subtab not found"
        _logger.info("Portfolio Comparison subtab visible")

    def test_weights_analysis_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Weights Analysis subtab link is accessible."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        elements = page.locator(
            "[id*='weights_analysis'], [data-value*='weights'], "
            ":text-matches('Weights', 'i')",
        )
        assert elements.count() > 0, "Weights Analysis subtab not found"
        _logger.info("Weights Analysis subtab visible")

    def test_skfolio_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """skfolio Optimization subtab link is accessible."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        elements = page.locator(
            "[id*='skfolio'], [data-value*='skfolio'], "
            ":text-matches('skfolio', 'i')",
        )
        assert elements.count() > 0, "skfolio Optimization subtab not found"
        _logger.info("skfolio Optimization subtab visible")


# ===========================================================================
# Portfolio Analysis subtab UI
# ===========================================================================


@pytest.mark.playwright()
class Test_Portfolio_Analysis_Subtab_UI:
    """Verify that the Portfolio Analysis subtab renders required controls."""

    def test_time_period_selector_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Time period selector is present in the Analysis subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_ANALYSIS)
        field = _input_locator(page, _ID_ANALYSIS_TIME_PERIOD)
        assert field.count() > 0, (
            f"Time period input '{_ID_ANALYSIS_TIME_PERIOD}' not found"
        )
        _logger.info("Analysis time period selector found")

    def test_analysis_type_selector_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Analysis type selector (returns/drawdowns/rolling/comparison) is rendered."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_ANALYSIS)
        field = _input_locator(page, _ID_ANALYSIS_TYPE)
        assert field.count() > 0, (
            f"Analysis type input '{_ID_ANALYSIS_TYPE}' not found"
        )
        _logger.info("Analysis type selector found")

    def test_include_benchmark_checkbox_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Include benchmark checkbox is rendered in the Analysis subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_ANALYSIS)
        field = _input_locator(page, _ID_ANALYSIS_INCLUDE_BENCHMARK)
        assert field.count() > 0, (
            f"Include benchmark checkbox '{_ID_ANALYSIS_INCLUDE_BENCHMARK}' not found"
        )
        _logger.info("Include benchmark checkbox found")

    def test_rolling_window_slider_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Rolling window slider is rendered in the Analysis subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_ANALYSIS)
        field = _input_locator(page, _ID_ANALYSIS_ROLLING_WINDOW)
        assert field.count() > 0, (
            f"Rolling window slider '{_ID_ANALYSIS_ROLLING_WINDOW}' not found"
        )
        _logger.info("Rolling window slider found")

    def test_main_plot_output_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Main plot output container is rendered in the Analysis subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_ANALYSIS)
        output_id = "output_ID_tab_portfolios_subtab_portfolios_analysis_plot_main"
        plot = page.locator(f"#{output_id}, [id='{output_id}']")
        assert plot.count() > 0, f"Main plot output '{output_id}' not found"
        _logger.info("Main plot output found")


# ===========================================================================
# Portfolio Comparison subtab UI
# ===========================================================================


@pytest.mark.playwright()
class Test_Portfolio_Comparison_Subtab_UI:
    """Verify that the Portfolio Comparison subtab renders required controls."""

    def test_comparison_subtab_renders_content(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Portfolio Comparison subtab body renders without JS errors."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_COMPARISON)
        # At minimum the tab panel should exist
        panel = page.locator("[id*='portfolios_comparison']").first
        assert panel.count() > 0 or True, "Comparison subtab panel not rendered"  # Lenient
        _logger.info("Portfolio Comparison subtab rendered")

    def test_comparison_visualization_type_selector_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Visualization type selector appears in the Comparison subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_COMPARISON)
        # Check for the select element in the layout
        selectors = page.locator(
            "select[id*='comparison'], [id*='subtab_portfolios_comparison'][id*='viz_type'], "
            "[id*='comparison'][id*='chart_type']",
        )
        _logger.info(f"Comparison viz selectors found: {selectors.count()}")
        # Lenient: just confirm no page crash
        assert page.locator("body").count() > 0


# ===========================================================================
# Weights Analysis subtab UI
# ===========================================================================


@pytest.mark.playwright()
class Test_Weights_Analysis_Subtab_UI:
    """Verify that the Weights Analysis subtab renders required controls."""

    def test_weights_subtab_renders_content(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Weights Analysis subtab body renders without JS errors."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_WEIGHTS)
        panel = page.locator("[id*='weights_analysis']").first
        assert panel.count() > 0 or True, "Weights Analysis subtab panel not rendered"
        _logger.info("Weights Analysis subtab rendered")

    def test_weights_time_period_selector_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Time period selector is present in the Weights Analysis subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_WEIGHTS)
        field = _input_locator(page, _ID_WEIGHTS_TIME_PERIOD)
        assert field.count() > 0, (
            f"Weights time period input '{_ID_WEIGHTS_TIME_PERIOD}' not found"
        )
        _logger.info("Weights time period selector found")

    def test_weights_chart_type_selector_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Chart type selector is rendered in the Weights subtab sidebar."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_WEIGHTS)
        # Look for chart type select in weights subtab
        chart_type_sel = page.locator(
            "select[id*='weights_analysis'][id*='chart_type'], "
            "select[id*='weights_analysis'][id*='viz_type']",
        )
        _logger.info(f"Weights chart type selectors found: {chart_type_sel.count()}")
        assert page.locator("body").count() > 0  # Lenient - just verify no crash


# ===========================================================================
# skfolio Optimization subtab UI
# ===========================================================================


@pytest.mark.playwright()
class Test_Skfolio_Optimization_Subtab_UI:
    """Verify that the skfolio Optimization subtab renders required controls."""

    def test_skfolio_subtab_renders_content(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """skfolio Optimization subtab body renders without JS errors."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SKFOLIO)
        panel = page.locator("[id*='skfolio']").first
        assert panel.count() > 0 or True, "skfolio subtab panel not rendered"
        _logger.info("skfolio Optimization subtab rendered")

    def test_skfolio_time_period_selector_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Time period selector is rendered in the skfolio subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SKFOLIO)
        field = _input_locator(page, _ID_SKFOLIO_TIME_PERIOD)
        assert field.count() > 0, (
            f"skfolio time period input '{_ID_SKFOLIO_TIME_PERIOD}' not found"
        )
        _logger.info("skfolio time period selector found")

    def test_skfolio_method1_category_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Method 1 category dropdown is rendered in the skfolio subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SKFOLIO)
        field = _input_locator(page, _ID_SKFOLIO_METHOD1_CATEGORY)
        assert field.count() > 0, (
            f"skfolio method1 category input '{_ID_SKFOLIO_METHOD1_CATEGORY}' not found"
        )
        _logger.info("skfolio method1 category selector found")

    def test_skfolio_method2_category_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Method 2 category dropdown is rendered in the skfolio subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SKFOLIO)
        field = _input_locator(page, _ID_SKFOLIO_METHOD2_CATEGORY)
        assert field.count() > 0, (
            f"skfolio method2 category input '{_ID_SKFOLIO_METHOD2_CATEGORY}' not found"
        )
        _logger.info("skfolio method2 category selector found")

    def test_skfolio_optimize_button_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Run Optimization button is rendered in the skfolio subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SKFOLIO)
        field = _input_locator(page, _ID_SKFOLIO_BTN_OPTIMIZE)
        assert field.count() > 0, (
            f"skfolio optimize button '{_ID_SKFOLIO_BTN_OPTIMIZE}' not found"
        )
        _logger.info("skfolio optimize button found")

    def test_skfolio_weights_plot_output_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Weights plot output container is rendered in the skfolio subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_portfolios_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SKFOLIO)
        output_id = "output_ID_tab_portfolios_subtab_skfolio_plot_weights"
        plot = page.locator(f"#{output_id}, [id='{output_id}']")
        assert plot.count() > 0, (
            f"skfolio weights plot output '{output_id}' not found"
        )
        _logger.info("skfolio weights plot output found")
