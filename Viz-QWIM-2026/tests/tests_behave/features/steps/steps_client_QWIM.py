"""Behave step definitions for Client_QWIM.

Tests cover:
    - Client construction with Client_Type enum (primary and partner)
    - update_personal_info / get_personal_info / age / risk tolerance accessors
    - update_assets / get_total_assets
    - update_goals / get_goals
    - update_income

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import sys
from pathlib import Path

from behave import given, then, when

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Module-level import guard
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    import polars as pl
    from src.clients_QWIM.client_QWIM import (
        Client_QWIM,
        Client_Type,
        Marital_Status,
    )
except ImportError as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)


def _require_imports() -> None:
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Client_QWIM source modules could not be imported: {_import_error_message}"
        )


def _make_personal_info_data(
    first_name: str,
    last_name: str,
    current_age: int,
    retirement_age: int,
    risk_tolerance: int,
) -> dict:
    """Build the personal info dict expected by update_personal_info."""
    return {
        "First Name": first_name,
        "Last Name": last_name,
        "Current Age": current_age,
        "Retirement Age": retirement_age,
        "Risk Tolerance": risk_tolerance,
        "Income Start Age": retirement_age,
    }


# ===========================================================================
# Given / When — construction
# ===========================================================================


@when(u'I create a primary client with id "{client_id}" first name "{first_name}" last name "{last_name}"')
def step_create_primary_client(context, client_id: str, first_name: str, last_name: str) -> None:
    """Create a Client_QWIM with CLIENT_PRIMARY type."""
    _require_imports()
    context.client = Client_QWIM(
        client_ID=client_id,
        first_name=first_name,
        last_name=last_name,
        client_type=Client_Type.CLIENT_PRIMARY,
    )


@when(u'I create a partner client with id "{client_id}" first name "{first_name}" last name "{last_name}"')
def step_create_partner_client(context, client_id: str, first_name: str, last_name: str) -> None:
    """Create a Client_QWIM with CLIENT_PARTNER type."""
    _require_imports()
    context.client2 = Client_QWIM(
        client_ID=client_id,
        first_name=first_name,
        last_name=last_name,
        client_type=Client_Type.CLIENT_PARTNER,
    )


@given(u'a primary client with id "{client_id}" first name "{first_name}" last name "{last_name}"')
def step_given_primary_client(context, client_id: str, first_name: str, last_name: str) -> None:
    """Pre-create a primary client for use as precondition."""
    _require_imports()
    context.client = Client_QWIM(
        client_ID=client_id,
        first_name=first_name,
        last_name=last_name,
        client_type=Client_Type.CLIENT_PRIMARY,
    )


# ===========================================================================
# Then — construction assertions
# ===========================================================================


@then(u'the client should be created without error')
def step_client_created(context) -> None:
    client = context.client if context.client is not None else context.client2
    assert client is not None, "Client was not created (is None)"


@then(u'the two clients should be different objects')
def step_clients_distinct(context) -> None:
    assert context.client is not context.client2, (
        "Expected two distinct client objects but got the same instance"
    )


# ===========================================================================
# When — personal info
# ===========================================================================


@when(u'I update personal info with current age {current_age:d} retirement age {retirement_age:d} and risk tolerance {risk_tolerance:d}')
def step_update_personal_info(
    context,
    current_age: int,
    retirement_age: int,
    risk_tolerance: int,
) -> None:
    """Call update_personal_info on context.client with supplied values."""
    _require_imports()
    data = _make_personal_info_data(
        first_name=context.client.m_first_name,
        last_name=context.client.m_last_name,
        current_age=current_age,
        retirement_age=retirement_age,
        risk_tolerance=risk_tolerance,
    )
    context.update_result = context.client.update_personal_info(data)


# ===========================================================================
# Then — personal info assertions
# ===========================================================================


@then(u'the personal info update should return True')
def step_personal_info_update_true(context) -> None:
    assert context.update_result is True, (
        f"update_personal_info returned {context.update_result!r}; expected True"
    )


@then(u'the stored current age should equal {expected_age:d}')
def step_current_age_equals(context, expected_age: int) -> None:
    actual = context.client.get_current_age()
    assert actual == expected_age, (
        f"Expected current age {expected_age} but got {actual}"
    )


@then(u'the stored retirement age should equal {expected_age:d}')
def step_retirement_age_equals(context, expected_age: int) -> None:
    actual = context.client.get_retirement_age()
    assert actual == expected_age, (
        f"Expected retirement age {expected_age} but got {actual}"
    )


@then(u'the stored risk tolerance should equal {expected_risk:d}')
def step_risk_tolerance_equals(context, expected_risk: int) -> None:
    actual = context.client.get_risk_tolerance()
    assert actual == expected_risk, (
        f"Expected risk tolerance {expected_risk} but got {actual}"
    )


@then(u'the retirement age should be greater than the current age')
def step_retirement_age_gt_current_age(context) -> None:
    current = context.client.get_current_age()
    retirement = context.client.get_retirement_age()
    assert retirement > current, (
        f"Expected retirement_age ({retirement}) > current_age ({current})"
    )


@then(u'the personal info DataFrame should not be empty')
def step_personal_info_df_not_empty(context) -> None:
    _require_imports()
    df = context.client.get_personal_info()
    assert isinstance(df, pl.DataFrame), (
        f"Expected pl.DataFrame for personal info, got {type(df)}"
    )
    assert len(df) > 0, "Personal info DataFrame is empty after update"


# ===========================================================================
# When — assets
# ===========================================================================


@when(u'I update assets with retirement account {retirement:g} and stocks {stocks:g}')
def step_update_assets(context, retirement: float, stocks: float) -> None:
    """Call update_assets with retirement (tax-deferred) and stocks (taxable)."""
    _require_imports()
    data: list[dict] = [
        {
            "Taxable Assets": float(stocks),
            "Tax Deferred Assets": float(retirement),
            "Tax Free Assets": 0.0,
            "Asset Name": "Test Asset",
            "Asset Class": "Mixed",
        }
    ]
    context.update_result = context.client.update_assets(data)
    context.expected_total_assets = float(retirement) + float(stocks)


# ===========================================================================
# Then — assets assertions
# ===========================================================================


@then(u'the assets update should return True')
def step_assets_update_true(context) -> None:
    assert context.update_result is True, (
        f"update_assets returned {context.update_result!r}; expected True"
    )


@then(u'the total assets should equal {expected:g}')
def step_total_assets_equal(context, expected: float) -> None:
    actual = context.client.get_total_assets()
    assert abs(actual - float(expected)) <= 0.01, (
        f"Expected total assets {expected} but got {actual}"
    )


@then(u'the total assets should be positive')
def step_total_assets_positive(context) -> None:
    total = context.client.get_total_assets()
    assert total > 0, f"Expected positive total assets but got {total}"


# ===========================================================================
# When — goals
# ===========================================================================


@when(u'I update goals with a retirement goal of {amount:g} by year {year:d}')
def step_update_goals(context, amount: float, year: int) -> None:
    """Call update_goals using essential expense derived from retirement target."""
    _require_imports()
    # Treat amount as an annual essential expense for the goal
    annual_essential = min(float(amount), 200000.0)
    data: list[dict] = [
        {
            "Essential Annual Expense": annual_essential,
            "Important Annual Expense": 20000.0,
            "Aspirational Annual Expense": 10000.0,
            "Essential Annual Expense is Inflation Indexed": True,
            "Important Annual Expense is Inflation Indexed": False,
            "Aspirational Annual Expense is Inflation Indexed": False,
        }
    ]
    context.update_result = context.client.update_goals(data)


# ===========================================================================
# Then — goals assertions
# ===========================================================================


@then(u'the goals update should return True')
def step_goals_update_true(context) -> None:
    assert context.update_result is True, (
        f"update_goals returned {context.update_result!r}; expected True"
    )


@then(u'the goals DataFrame should not be empty')
def step_goals_df_not_empty(context) -> None:
    _require_imports()
    df = context.client.get_goals()
    assert isinstance(df, pl.DataFrame), (
        f"Expected pl.DataFrame for goals, got {type(df)}"
    )
    assert len(df) > 0, "Goals DataFrame is empty after update"


# ===========================================================================
# When — income
# ===========================================================================


@when(u'I update income with salary {salary:g} and social security {social_security:g}')
def step_update_income(context, salary: float, social_security: float) -> None:
    """Call update_income with social security and other income entries."""
    _require_imports()
    data: list[dict] = [
        {
            "Annual Social Security": float(social_security),
            "Annual Income from Pension": 0.0,
            "Annual Income from Existing Annuity": 0.0,
            "Annual Income from Other Sources": float(salary),
            "Annual Income from Pension is Inflation Indexed": False,
            "Annual Income from Existing Annuity is Inflation Indexed": False,
            "Annual Income from Other Sources is Inflation Indexed": False,
            "Income Start Age for Pension": 65,
            "Income Start Age for Existing Annuity": 65,
            "Income Start Age for Other Sources": 65,
            "Income Duration for Pension": 25,
            "Income Duration for Existing Annuity": 25,
            "Income Duration for Other Sources": 25,
        }
    ]
    context.update_result = context.client.update_income(data)


# ===========================================================================
# Then — income assertions
# ===========================================================================


@then(u'the income update should return True')
def step_income_update_true(context) -> None:
    assert context.update_result is True, (
        f"update_income returned {context.update_result!r}; expected True"
    )
