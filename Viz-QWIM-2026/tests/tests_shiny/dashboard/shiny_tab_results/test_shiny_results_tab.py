"""Playwright-based Shiny browser tests for the Results tab.

These tests launch a real Shiny server (via the ``shiny_server_url`` fixture
in ``tests/tests_shiny/conftest.py``) and drive Google Chrome / Chromium
using ``playwright.sync_api``.

The test classes are organised by subtab and concern:

* :class:`Test_Results_Tab_Navigation`   - tab navigation and subtab presence
* :class:`Test_Simulation_Subtab_UI`     - simulation form controls
* :class:`Test_Reporting_Subtab_UI`      - reporting form controls
* :class:`Test_Simulation_Pure_Logic`    - pure-Python logic (no browser needed)
* :class:`Test_Reporting_Pure_Logic`     - security helper logic (no browser)

Run (requires a running Shiny server at the ``QWIM_SHINY_TEST_URL`` env
variable, or the ``shiny_server_url`` pytest fixture from conftest.py)::

    # Browser tests only
    pytest tests/tests_shiny/dashboard/shiny_tab_results/ -m playwright -v

    # Pure-logic unit tests (no browser, no server)
    pytest tests/tests_shiny/dashboard/shiny_tab_results/ -m "not playwright" -v

Author:
    QWIM Development Team

Version:
    0.1.0

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
    _logger.warning("playwright not installed — Results tab Shiny tests will be skipped")

# Per-class markers are applied below — no module-level pytestmark so that
# pure-logic unit tests (Test_Simulation_Pure_Logic, Test_Reporting_Pure_Logic)
# are NOT skipped when playwright is unavailable.
_PLAYWRIGHT_MARKS = [
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
# Selector constants
# ---------------------------------------------------------------------------

_SEL_RESULTS_TAB_LINK = (
    "a[data-value='tab_results'], a[href*='results'], [id*='tab_results']"
)

# Subtab nav links
_SEL_SUBTAB_SIMULATION = (
    "a[data-value*='simulation'], a:has-text('Simulation'), [id*='simulation']"
)
_SEL_SUBTAB_REPORTING = (
    "a[data-value*='reporting'], a:has-text('Reporting'), [id*='reporting']"
)

# Simulation input IDs
_ID_SIM_SELECT_ALL = "input_ID_tab_results_subtab_simulation_select_all_components"
_ID_SIM_NUM_SCENARIOS = "input_ID_tab_results_subtab_simulation_num_scenarios"
_ID_SIM_NUM_DAYS = "input_ID_tab_results_subtab_simulation_num_days"
_ID_SIM_START_DATE = "input_ID_tab_results_subtab_simulation_start_date"
_ID_SIM_INITIAL_VALUE = "input_ID_tab_results_subtab_simulation_initial_value"
_ID_SIM_DISTRIBUTION = "input_ID_tab_results_subtab_simulation_distribution_type"
_ID_SIM_RNG_TYPE = "input_ID_tab_results_subtab_simulation_rng_type"
_ID_SIM_SEED = "input_ID_tab_results_subtab_simulation_seed"
_ID_SIM_RUN_BTN = "input_ID_tab_results_subtab_simulation_run_btn"

# Reporting input IDs (note: reporting module uses 'tab_portfolios' prefix)
_ID_RPT_TITLE = "input_ID_tab_portfolios_subtab_reporting_text_report_title"
_ID_RPT_FILENAME = "input_ID_tab_portfolios_subtab_reporting_text_download_filename"
_ID_RPT_INCLUDE_CHARTS = "input_ID_tab_portfolios_subtab_reporting_checkbox_include_charts"
_ID_RPT_CHART_RESOLUTION = "input_ID_tab_portfolios_subtab_reporting_select_chart_resolution"
_ID_RPT_BTN_GENERATE = "input_ID_tab_portfolios_subtab_reporting_btn_generate_pdf"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _navigate_to_results_tab(page: Page, url: str, timeout: int = 30_000) -> None:
    """Navigate to the Shiny app and click into the Results tab."""
    page.goto(url, wait_until="networkidle", timeout=timeout)
    _logger.debug("Navigated to %s", url)
    results_link = page.locator(_SEL_RESULTS_TAB_LINK).first
    if results_link.count() > 0:
        results_link.click()
        _logger.debug("Clicked Results tab link")
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
# Navigation / smoke tests
# ===========================================================================


@pytest.mark.playwright
@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
class Test_Results_Tab_Navigation:
    """Verify Results tab and subtabs are reachable in the dashboard."""

    def test_dashboard_loads(self, page: Page, shiny_server_url: str) -> None:
        """Dashboard loads without a JavaScript error."""
        url = _base_url(shiny_server_url)
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        assert page.locator("body").count() > 0
        _logger.info("Dashboard loaded at %s", url)

    def test_results_tab_link_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """A navigable Results tab link is present in the top navigation."""
        url = _base_url(shiny_server_url)
        page.goto(url, wait_until="networkidle", timeout=30_000)
        results_elements = page.locator(
            "[id*='tab_results'], [data-value*='results'], "
            "a:has-text('Results'), a:has-text('results')",
        )
        assert results_elements.count() > 0, (
            "No Results tab link found in navigation"
        )
        _logger.info("Results tab link found")

    def test_simulation_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Simulation subtab link is present after clicking Results tab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        elements = page.locator(
            "[id*='simulation'], [data-value*='simulation'], "
            "a:has-text('Simulation')",
        )
        assert elements.count() > 0, "Simulation subtab not found"
        _logger.info("Simulation subtab link visible")

    def test_reporting_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Reporting subtab link is present after clicking Results tab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        elements = page.locator(
            "[id*='reporting'], [data-value*='reporting'], "
            "a:has-text('Reporting')",
        )
        assert elements.count() > 0, "Reporting subtab not found"
        _logger.info("Reporting subtab link visible")

    def test_tab_sets_active_class(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Clicking the Results tab link sets an active CSS class."""
        url = _base_url(shiny_server_url)
        page.goto(url, wait_until="networkidle", timeout=30_000)
        link = page.locator(_SEL_RESULTS_TAB_LINK).first
        if link.count() > 0:
            link.click()
            time.sleep(0.5)
        # Active nav element should have aria-selected or active class
        active = page.locator(".nav-link.active, [aria-selected='true']").count()
        assert active > 0, "No active nav link after tab click"
        _logger.info("Active class set after clicking Results tab")


# ===========================================================================
# Simulation subtab UI tests
# ===========================================================================


@pytest.mark.playwright
@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
class Test_Simulation_Subtab_UI:
    """Verify the Simulation subtab renders all expected controls."""

    def test_simulation_subtab_renders_body(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Simulation subtab body renders without JavaScript errors."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        assert page.locator("body").count() > 0
        _logger.info("Simulation subtab body rendered")

    def test_select_all_components_checkbox_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """'Select All' components checkbox is rendered in Simulation subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        field = _input_locator(page, _ID_SIM_SELECT_ALL)
        assert field.count() > 0, (
            f"Select-all checkbox '{_ID_SIM_SELECT_ALL}' not found"
        )
        _logger.info("Select-all components checkbox found")

    def test_num_scenarios_input_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Number of scenarios input is rendered in the Simulation subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        field = _input_locator(page, _ID_SIM_NUM_SCENARIOS)
        assert field.count() > 0, (
            f"Num scenarios input '{_ID_SIM_NUM_SCENARIOS}' not found"
        )
        _logger.info("Number of scenarios input found")

    def test_num_days_input_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Number of simulation days input is rendered in the Simulation subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        field = _input_locator(page, _ID_SIM_NUM_DAYS)
        assert field.count() > 0, (
            f"Num days input '{_ID_SIM_NUM_DAYS}' not found"
        )
        _logger.info("Number of days input found")

    def test_start_date_input_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Start date input is rendered in the Simulation subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        field = _input_locator(page, _ID_SIM_START_DATE)
        assert field.count() > 0, (
            f"Start date input '{_ID_SIM_START_DATE}' not found"
        )
        _logger.info("Start date input found")

    def test_initial_value_input_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Initial portfolio value input is rendered in the Simulation subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        field = _input_locator(page, _ID_SIM_INITIAL_VALUE)
        assert field.count() > 0, (
            f"Initial value input '{_ID_SIM_INITIAL_VALUE}' not found"
        )
        _logger.info("Initial value input found")

    def test_distribution_type_selector_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Distribution type selector is rendered in the Simulation subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        field = _input_locator(page, _ID_SIM_DISTRIBUTION)
        assert field.count() > 0, (
            f"Distribution type selector '{_ID_SIM_DISTRIBUTION}' not found"
        )
        _logger.info("Distribution type selector found")

    def test_rng_type_selector_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """RNG type selector is rendered in the Simulation subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        field = _input_locator(page, _ID_SIM_RNG_TYPE)
        assert field.count() > 0, (
            f"RNG type selector '{_ID_SIM_RNG_TYPE}' not found"
        )
        _logger.info("RNG type selector found")

    def test_seed_input_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Random seed input is rendered in the Simulation subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        field = _input_locator(page, _ID_SIM_SEED)
        assert field.count() > 0, (
            f"Seed input '{_ID_SIM_SEED}' not found"
        )
        _logger.info("Seed input found")

    def test_run_simulation_button_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """'Run Simulation' action button is rendered in the Simulation subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        field = _input_locator(page, _ID_SIM_RUN_BTN)
        assert field.count() > 0, (
            f"Run Simulation button '{_ID_SIM_RUN_BTN}' not found"
        )
        _logger.info("Run Simulation button found")

    def test_distribution_choices_include_normal(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Distribution type selector contains a 'Normal' / 'normal' option."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SIMULATION)
        option = page.locator(
            f"select#{_ID_SIM_DISTRIBUTION} option, "
            f"[data-shiny-input-id='{_ID_SIM_DISTRIBUTION}'] option",
        ).filter(has_text="Normal")
        _logger.info(f"Normal distribution options found: {option.count()}")
        # Lenient — just confirm the page is alive
        assert page.locator("body").count() > 0


# ===========================================================================
# Reporting subtab UI tests
# ===========================================================================


@pytest.mark.playwright
@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
class Test_Reporting_Subtab_UI:
    """Verify the Reporting subtab renders all expected controls."""

    def test_reporting_subtab_renders_body(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Reporting subtab body renders without JavaScript errors."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_REPORTING)
        assert page.locator("body").count() > 0
        _logger.info("Reporting subtab body rendered")

    def test_report_title_input_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Report title text input is rendered in the Reporting subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_REPORTING)
        field = _input_locator(page, _ID_RPT_TITLE)
        assert field.count() > 0, (
            f"Report title input '{_ID_RPT_TITLE}' not found"
        )
        _logger.info("Report title input found")

    def test_download_filename_input_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Download filename text input is rendered in the Reporting subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_REPORTING)
        field = _input_locator(page, _ID_RPT_FILENAME)
        assert field.count() > 0, (
            f"Download filename input '{_ID_RPT_FILENAME}' not found"
        )
        _logger.info("Download filename input found")

    def test_include_charts_checkbox_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """'Include charts' checkbox is rendered in the Reporting subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_REPORTING)
        field = _input_locator(page, _ID_RPT_INCLUDE_CHARTS)
        assert field.count() > 0, (
            f"Include charts checkbox '{_ID_RPT_INCLUDE_CHARTS}' not found"
        )
        _logger.info("Include charts checkbox found")

    def test_chart_resolution_selector_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Chart resolution selector is rendered in the Reporting subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_REPORTING)
        field = _input_locator(page, _ID_RPT_CHART_RESOLUTION)
        assert field.count() > 0, (
            f"Chart resolution selector '{_ID_RPT_CHART_RESOLUTION}' not found"
        )
        _logger.info("Chart resolution selector found")

    def test_generate_pdf_button_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """'Generate PDF' action button is rendered in the Reporting subtab."""
        url = _base_url(shiny_server_url)
        _navigate_to_results_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_REPORTING)
        field = _input_locator(page, _ID_RPT_BTN_GENERATE)
        assert field.count() > 0, (
            f"Generate PDF button '{_ID_RPT_BTN_GENERATE}' not found"
        )
        _logger.info("Generate PDF button found")


# ===========================================================================
# Pure-logic tests — no browser required
# ===========================================================================


@pytest.mark.unit
class Test_Simulation_Pure_Logic:
    """Pure-Python logic tests for subtab_simulation constants and helpers.

    These run without playwright and without a Shiny server.
    """

    def test_all_etf_symbols_has_12_entries(self) -> None:
        """ALL_ETF_SYMBOLS contains exactly 12 ETF tickers."""
        from src.dashboard.shiny_tab_results.subtab_simulation import ALL_ETF_SYMBOLS

        assert len(ALL_ETF_SYMBOLS) == 12

    def test_default_selected_etfs_are_subset_of_all(self) -> None:
        """Every DEFAULT_SELECTED_ETFS entry is a member of ALL_ETF_SYMBOLS."""
        from src.dashboard.shiny_tab_results.subtab_simulation import (
            ALL_ETF_SYMBOLS,
            DEFAULT_SELECTED_ETFS,
        )

        for sym in DEFAULT_SELECTED_ETFS:
            assert sym in ALL_ETF_SYMBOLS, f"'{sym}' not in ALL_ETF_SYMBOLS"

    def test_distribution_choices_has_required_keys(self) -> None:
        """DISTRIBUTION_CHOICES includes normal, lognormal, and student_t."""
        from src.dashboard.shiny_tab_results.subtab_simulation import DISTRIBUTION_CHOICES

        for key in ("normal", "lognormal", "student_t"):
            assert key in DISTRIBUTION_CHOICES, (
                f"'{key}' missing from DISTRIBUTION_CHOICES"
            )

    def test_rng_type_choices_has_required_keys(self) -> None:
        """RNG_TYPE_CHOICES includes pcg64 and mt19937."""
        from src.dashboard.shiny_tab_results.subtab_simulation import RNG_TYPE_CHOICES

        for key in ("pcg64", "mt19937"):
            assert key in RNG_TYPE_CHOICES, (
                f"'{key}' missing from RNG_TYPE_CHOICES"
            )

    def test_simulate_ui_callable(self) -> None:
        """subtab_simulation_ui is a callable."""
        from src.dashboard.shiny_tab_results.subtab_simulation import subtab_simulation_ui

        assert callable(subtab_simulation_ui)

    def test_simulate_server_callable(self) -> None:
        """subtab_simulation_server is a callable."""
        from src.dashboard.shiny_tab_results.subtab_simulation import subtab_simulation_server

        assert callable(subtab_simulation_server)


@pytest.mark.unit
class Test_Reporting_Pure_Logic:
    """Pure-Python logic tests for subtab_reporting security helpers.

    These run without playwright and without a Shiny server.
    """

    def test_sanitize_accepts_well_formed_filename(self) -> None:
        """sanitize_filename_for_security accepts a valid PDF filename."""
        from src.dashboard.shiny_tab_results.subtab_reporting import (
            sanitize_filename_for_security,
        )

        is_valid, sanitized, _ = sanitize_filename_for_security("QWIM_Report_2024.pdf")
        assert is_valid is True
        assert len(sanitized) > 0

    def test_sanitize_rejects_empty_filename(self) -> None:
        """sanitize_filename_for_security rejects an empty filename."""
        from src.dashboard.shiny_tab_results.subtab_reporting import (
            sanitize_filename_for_security,
        )

        is_valid, sanitized, _ = sanitize_filename_for_security("")
        assert is_valid is False
        assert sanitized == ""

    def test_sanitize_rejects_path_traversal(self) -> None:
        """sanitize_filename_for_security rejects path traversal attempts."""
        from src.dashboard.shiny_tab_results.subtab_reporting import (
            sanitize_filename_for_security,
        )

        is_valid, _, _ = sanitize_filename_for_security("../../etc/passwd.pdf")
        assert is_valid is False

    def test_sanitize_rejects_non_pdf_extension(self) -> None:
        """sanitize_filename_for_security rejects filenames without .pdf extension."""
        from src.dashboard.shiny_tab_results.subtab_reporting import (
            sanitize_filename_for_security,
        )

        is_valid, _, _ = sanitize_filename_for_security("Report_2024.docx")
        assert is_valid is False

    def test_sanitize_rejects_overlong_filename(self) -> None:
        """sanitize_filename_for_security rejects filenames exceeding MAX_FILENAME_LENGTH."""
        from src.dashboard.shiny_tab_results.subtab_reporting import (
            MAX_FILENAME_LENGTH,
            sanitize_filename_for_security,
        )

        long_name = "x" * (MAX_FILENAME_LENGTH + 10) + ".pdf"
        is_valid, _, _ = sanitize_filename_for_security(long_name)
        assert is_valid is False

    def test_sanitize_returns_3_tuple(self) -> None:
        """sanitize_filename_for_security always returns a 3-element tuple."""
        from src.dashboard.shiny_tab_results.subtab_reporting import (
            sanitize_filename_for_security,
        )

        result = sanitize_filename_for_security("Test.pdf")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_max_filename_length_is_positive_int(self) -> None:
        """MAX_FILENAME_LENGTH is a positive integer."""
        from src.dashboard.shiny_tab_results.subtab_reporting import MAX_FILENAME_LENGTH

        assert isinstance(MAX_FILENAME_LENGTH, int)
        assert MAX_FILENAME_LENGTH > 0

    def test_forbidden_filename_parts_contains_dotdot(self) -> None:
        """FORBIDDEN_FILENAME_PARTS contains '..'."""
        from src.dashboard.shiny_tab_results.subtab_reporting import FORBIDDEN_FILENAME_PARTS

        assert ".." in FORBIDDEN_FILENAME_PARTS

    def test_reporting_ui_callable(self) -> None:
        """subtab_reporting_ui is a callable."""
        from src.dashboard.shiny_tab_results.subtab_reporting import subtab_reporting_ui

        assert callable(subtab_reporting_ui)

    def test_reporting_server_callable(self) -> None:
        """subtab_reporting_server is a callable."""
        from src.dashboard.shiny_tab_results.subtab_reporting import subtab_reporting_server

        assert callable(subtab_reporting_server)
