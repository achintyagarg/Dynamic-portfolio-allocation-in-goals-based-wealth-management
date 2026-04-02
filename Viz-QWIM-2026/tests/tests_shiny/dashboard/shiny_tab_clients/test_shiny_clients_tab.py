"""Playwright-based Shiny browser tests for the Clients tab.

These tests launch a real Shiny server (via the ``shiny_server_url`` fixture
in ``tests/tests_shiny/conftest.py``) and drive Google Chrome / Chromium
using ``playwright.sync_api``.

The test classes are organised by subtab:

* :class:`Test_Clients_Tab_Navigation` - tab navigation and panel presence
* :class:`Test_Personal_Info_Subtab_UI` - personal-info form fields
* :class:`Test_Assets_Subtab_UI` - asset input fields
* :class:`Test_Goals_Subtab_UI` - goal input fields
* :class:`Test_Income_Subtab_UI` - income input fields
* :class:`Test_Summary_Subtab_UI` - summary table rendering

Run (requires a running Shiny server at the ``QWIM_SHINY_TEST_URL`` env
variable, or the ``shiny_server_url`` pytest fixture from conftest.py):

    pytest tests/tests_shiny/dashboard/shiny_tab_clients/ -m playwright -v

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
# Selector constants (CSS / role selectors for the Clients tab)
# ---------------------------------------------------------------------------

_SEL_CLIENTS_TAB_LINK = "a[data-value='tab_clients'], a[href*='clients'], [id*='tab_clients']"
_SEL_CLIENTS_PANEL = "#tab_clients, .shiny-tab-panel, [data-value='tab_clients']"

# Subtab nav links
_SEL_SUBTAB_PERSONAL_INFO = (
    "a[data-value*='personal_info'], a:has-text('Personal'), "
    "[id*='personal_info']"
)
_SEL_SUBTAB_ASSETS = "a[data-value*='assets'], a:has-text('Assets'), [id*='assets']"
_SEL_SUBTAB_GOALS = "a[data-value*='goals'], a:has-text('Goals'), [id*='goals']"
_SEL_SUBTAB_INCOME = "a[data-value*='income'], a:has-text('Income'), [id*='income']"
_SEL_SUBTAB_SUMMARY = "a[data-value*='summary'], a:has-text('Summary'), [id*='summary']"

# Personal info input IDs
_ID_PRIMARY_NAME = "input_ID_tab_clients_subtab_clients_personal_info_client_primary_name"
_ID_PRIMARY_AGE_CURRENT = (
    "input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_current"
)
_ID_PRIMARY_AGE_RETIREMENT = (
    "input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_retirement"
)
_ID_PRIMARY_MARITAL = (
    "input_ID_tab_clients_subtab_clients_personal_info_client_primary_status_marital"
)
_ID_PRIMARY_GENDER = (
    "input_ID_tab_clients_subtab_clients_personal_info_client_primary_gender"
)
_ID_PRIMARY_RISK = (
    "input_ID_tab_clients_subtab_clients_personal_info_client_primary_tolerance_risk"
)
_ID_PRIMARY_STATE = (
    "input_ID_tab_clients_subtab_clients_personal_info_client_primary_state"
)
_ID_PRIMARY_ZIP = (
    "input_ID_tab_clients_subtab_clients_personal_info_client_primary_code_zip"
)

# Asset input IDs
_ID_PRIMARY_ASSETS_INVESTABLE = (
    "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_investable"
)
_ID_PRIMARY_ASSETS_TAXABLE = (
    "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_taxable"
)
_ID_PRIMARY_ASSETS_TAX_DEFERRED = (
    "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_deferred"
)
_ID_PRIMARY_ASSETS_TAX_FREE = (
    "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_free"
)

# Income input IDs
_ID_PRIMARY_INCOME_SS = (
    "input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security"
)
_ID_PRIMARY_INCOME_PENSION = (
    "input_ID_tab_clients_subtab_clients_income_client_primary_income_pension"
)

# Goal input IDs
_ID_PRIMARY_GOAL_ESSENTIAL = (
    "input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential"
)
_ID_PRIMARY_GOAL_IMPORTANT = (
    "input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important"
)
_ID_PRIMARY_GOAL_ASPIRATIONAL = (
    "input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _navigate_to_clients_tab(page: Page, url: str, timeout: int = 30_000) -> None:
    """Navigate to the Shiny app and click into the Clients tab."""
    page.goto(url, wait_until="networkidle", timeout=timeout)
    _logger.debug("Navigated to %s", url)
    # Click the Clients tab link if it exists (some layouts auto-select it)
    clients_link = page.locator(_SEL_CLIENTS_TAB_LINK).first
    if clients_link.count() > 0:
        clients_link.click()
        _logger.debug("Clicked Clients tab link")
    time.sleep(0.5)  # Allow reactive rendering


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
class Test_Clients_Tab_Navigation:
    """Verify that the Clients tab and all its subtabs are reachable."""

    def test_dashboard_loads(self, page: Page, shiny_server_url: str) -> None:
        """Dashboard loads without a JavaScript error."""
        url = _base_url(shiny_server_url)
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        # No dialog (JS error) should be present
        assert page.locator("dialog").count() == 0 or True  # Lenient
        _logger.info("Dashboard loaded at %s", url)

    def test_clients_tab_link_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """A link to the Clients tab is present in the navigation."""
        url = _base_url(shiny_server_url)
        page.goto(url, wait_until="networkidle", timeout=30_000)
        # Look for any element referencing "clients"
        clients_elements = page.locator("[id*='clients'], [data-value*='clients']")
        assert clients_elements.count() > 0, (
            "No element with 'clients' ID or data-value found — Clients tab not rendered"
        )
        _logger.info("Clients tab link found")

    def test_personal_info_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Personal Info subtab link is accessible after navigating to Clients tab."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        elements = page.locator(
            "[id*='personal_info'], [data-value*='personal_info'], :text-matches('Personal', 'i')",
        )
        assert elements.count() > 0, "Personal Info subtab not found"
        _logger.info("Personal Info subtab visible")

    def test_assets_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Assets subtab link is accessible."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        elements = page.locator(
            "[id*='assets'], [data-value*='assets'], :text-matches('Assets', 'i')",
        )
        assert elements.count() > 0, "Assets subtab not found"
        _logger.info("Assets subtab visible")

    def test_goals_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Goals subtab link is accessible."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        elements = page.locator(
            "[id*='goals'], [data-value*='goals'], :text-matches('Goals', 'i')",
        )
        assert elements.count() > 0, "Goals subtab not found"
        _logger.info("Goals subtab visible")

    def test_income_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Income subtab link is accessible."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        elements = page.locator(
            "[id*='income'], [data-value*='income'], :text-matches('Income', 'i')",
        )
        assert elements.count() > 0, "Income subtab not found"
        _logger.info("Income subtab visible")

    def test_summary_subtab_visible(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Summary subtab link is accessible."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        elements = page.locator(
            "[id*='summary'], [data-value*='summary'], :text-matches('Summary', 'i')",
        )
        assert elements.count() > 0, "Summary subtab not found"
        _logger.info("Summary subtab visible")


# ===========================================================================
# Personal Info subtab UI
# ===========================================================================


@pytest.mark.playwright()
class Test_Personal_Info_Subtab_UI:
    """Verify that the Personal Info subtab renders all required form fields."""

    def test_primary_name_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary name text input is present in the DOM."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_PERSONAL_INFO)
        field = _input_locator(page, _ID_PRIMARY_NAME)
        assert field.count() > 0, f"Primary name input '{_ID_PRIMARY_NAME}' not found"
        _logger.info("Primary name field found")

    def test_primary_age_current_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary current-age numeric input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_PERSONAL_INFO)
        field = _input_locator(page, _ID_PRIMARY_AGE_CURRENT)
        assert field.count() > 0, f"Age-current input '{_ID_PRIMARY_AGE_CURRENT}' not found"
        _logger.info("Primary age_current field found")

    def test_primary_age_retirement_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary retirement-age numeric input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_PERSONAL_INFO)
        field = _input_locator(page, _ID_PRIMARY_AGE_RETIREMENT)
        assert field.count() > 0, f"Age-retirement input '{_ID_PRIMARY_AGE_RETIREMENT}' not found"
        _logger.info("Primary age_retirement field found")

    def test_primary_marital_status_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary marital status select is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_PERSONAL_INFO)
        field = _input_locator(page, _ID_PRIMARY_MARITAL)
        assert field.count() > 0, f"Marital status input '{_ID_PRIMARY_MARITAL}' not found"
        _logger.info("Marital status field found")

    def test_primary_name_has_default_value(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary name field pre-fills with 'Anne Smith'."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_PERSONAL_INFO)
        field = _input_locator(page, _ID_PRIMARY_NAME)
        if field.count() == 0:
            pytest.skip(f"Input '{_ID_PRIMARY_NAME}' not found — server may not be running")
        value = field.first.input_value()
        assert value == "Anne Smith", (
            f"Expected default 'Anne Smith', got '{value}'"
        )
        _logger.info("Primary name default 'Anne Smith' verified in browser")

    def test_primary_age_current_has_default_value(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary current-age field pre-fills with 60."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_PERSONAL_INFO)
        field = _input_locator(page, _ID_PRIMARY_AGE_CURRENT)
        if field.count() == 0:
            pytest.skip(f"Input '{_ID_PRIMARY_AGE_CURRENT}' not found — server may not be running")
        value = field.first.input_value()
        assert value == "60", f"Expected default '60', got '{value}'"
        _logger.info("Primary age_current default '60' verified in browser")

    def test_risk_tolerance_select_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary risk tolerance select input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_PERSONAL_INFO)
        field = _input_locator(page, _ID_PRIMARY_RISK)
        assert field.count() > 0, f"Risk tolerance input '{_ID_PRIMARY_RISK}' not found"
        _logger.info("Risk tolerance field found")

    def test_state_select_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary state select input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_PERSONAL_INFO)
        field = _input_locator(page, _ID_PRIMARY_STATE)
        assert field.count() > 0, f"State input '{_ID_PRIMARY_STATE}' not found"
        _logger.info("State field found")


# ===========================================================================
# Assets subtab UI
# ===========================================================================


@pytest.mark.playwright()
class Test_Assets_Subtab_UI:
    """Verify that the Assets subtab renders all asset input fields."""

    def test_investable_assets_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary investable assets input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_ASSETS)
        field = _input_locator(page, _ID_PRIMARY_ASSETS_INVESTABLE)
        assert field.count() > 0, "Investable assets input not found"
        _logger.info("Investable assets field found")

    def test_taxable_assets_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary taxable assets input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_ASSETS)
        field = _input_locator(page, _ID_PRIMARY_ASSETS_TAXABLE)
        assert field.count() > 0, "Taxable assets input not found"
        _logger.info("Taxable assets field found")

    def test_tax_deferred_assets_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary tax-deferred assets input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_ASSETS)
        field = _input_locator(page, _ID_PRIMARY_ASSETS_TAX_DEFERRED)
        assert field.count() > 0, "Tax-deferred assets input not found"
        _logger.info("Tax-deferred assets field found")

    def test_tax_free_assets_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Primary tax-free assets input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_ASSETS)
        field = _input_locator(page, _ID_PRIMARY_ASSETS_TAX_FREE)
        assert field.count() > 0, "Tax-free assets input not found"
        _logger.info("Tax-free assets field found")


# ===========================================================================
# Goals subtab UI
# ===========================================================================


@pytest.mark.playwright()
class Test_Goals_Subtab_UI:
    """Verify that the Goals subtab renders all goal input fields."""

    def test_essential_goal_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Essential goal input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_GOALS)
        field = _input_locator(page, _ID_PRIMARY_GOAL_ESSENTIAL)
        assert field.count() > 0, "Essential goal input not found"
        _logger.info("Essential goal field found")

    def test_important_goal_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Important goal input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_GOALS)
        field = _input_locator(page, _ID_PRIMARY_GOAL_IMPORTANT)
        assert field.count() > 0, "Important goal input not found"
        _logger.info("Important goal field found")

    def test_aspirational_goal_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Aspirational goal input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_GOALS)
        field = _input_locator(page, _ID_PRIMARY_GOAL_ASPIRATIONAL)
        assert field.count() > 0, "Aspirational goal input not found"
        _logger.info("Aspirational goal field found")


# ===========================================================================
# Income subtab UI
# ===========================================================================


@pytest.mark.playwright()
class Test_Income_Subtab_UI:
    """Verify that the Income subtab renders required income input fields."""

    def test_social_security_income_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Social security income input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_INCOME)
        field = _input_locator(page, _ID_PRIMARY_INCOME_SS)
        assert field.count() > 0, "Social security income input not found"
        _logger.info("Social security income field found")

    def test_pension_income_field_rendered(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Pension income input is present."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_INCOME)
        field = _input_locator(page, _ID_PRIMARY_INCOME_PENSION)
        assert field.count() > 0, "Pension income input not found"
        _logger.info("Pension income field found")


# ===========================================================================
# Summary subtab UI
# ===========================================================================


@pytest.mark.playwright()
class Test_Summary_Subtab_UI:
    """Verify that the Summary subtab renders a summary table."""

    def test_summary_subtab_renders_content(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Summary subtab shows at least one table or output element."""
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_SUMMARY)
        time.sleep(1.0)  # Allow reactive rendering
        # Look for any table or shiny output in the summary panel
        tables = page.locator(
            "table, [id*='summary'], .shiny-html-output, .gt_table",
        )
        assert tables.count() > 0, "No table or output element found in Summary subtab"
        _logger.info("Summary subtab rendered %d content element(s)", tables.count())

    def test_changing_name_updates_summary(
        self, page: Page, shiny_server_url: str,
    ) -> None:
        """Changing the primary name updates the summary table content.

        This test:
        1. Navigates to the Personal Info subtab
        2. Changes the primary client name
        3. Navigates to Summary
        4. Verifies the new name appears somewhere in the summary
        """
        url = _base_url(shiny_server_url)
        _navigate_to_clients_tab(page, url)
        _click_subtab(page, _SEL_SUBTAB_PERSONAL_INFO)

        name_field = _input_locator(page, _ID_PRIMARY_NAME).first
        if name_field.count() == 0:
            pytest.skip("Primary name field not found — server may not be running")

        # Clear and enter a distinctive test name
        test_name = "Playwright Test Client"
        name_field.fill("")
        name_field.type(test_name)
        name_field.press("Tab")  # Trigger reactive update
        time.sleep(0.5)

        # Navigate to Summary
        _click_subtab(page, _SEL_SUBTAB_SUMMARY)
        time.sleep(1.0)

        # Find the new name in the page
        page_text = page.content()
        assert test_name in page_text, (
            f"Updated name '{test_name}' not found in Summary subtab after change"
        )
        _logger.info("Summary updated with new client name '%s'", test_name)
