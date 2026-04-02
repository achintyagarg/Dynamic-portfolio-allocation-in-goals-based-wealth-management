"""
Robot Framework keyword library for Client_QWIM.
=================================================

Tests cover:
    - Client construction with Client_Type enum
    - update_personal_info / get_personal_info / age / risk tolerance accessors
    - update_assets / get_assets / get_total_assets / get_taxable_assets
    - update_goals / get_goals
    - update_income / get_total_annual_income

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so src packages resolve correctly
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Conditional imports with Module availability guard
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
    """Raise RuntimeError when required imports are unavailable."""
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Required imports not available: {_import_error_message}"
        )


# ---------------------------------------------------------------------------
# Construction keywords
# ---------------------------------------------------------------------------


def create_primary_client(client_id: str, first_name: str, last_name: str) -> Client_QWIM:
    """Create a Client_QWIM with Client_Type.CLIENT_PRIMARY.

    Arguments:
    - client_id   -- unique string identifier
    - first_name  -- client's first name
    - last_name   -- client's last name

    Returns: Client_QWIM instance
    """
    _require_imports()
    return Client_QWIM(
        client_ID=client_id,
        first_name=first_name,
        last_name=last_name,
        client_type=Client_Type.CLIENT_PRIMARY,
    )


def create_partner_client(client_id: str, first_name: str, last_name: str) -> Client_QWIM:
    """Create a Client_QWIM with Client_Type.CLIENT_PARTNER.

    Arguments:
    - client_id   -- unique string identifier
    - first_name  -- client's first name
    - last_name   -- client's last name

    Returns: Client_QWIM instance
    """
    _require_imports()
    return Client_QWIM(
        client_ID=client_id,
        first_name=first_name,
        last_name=last_name,
        client_type=Client_Type.CLIENT_PARTNER,
    )


# ---------------------------------------------------------------------------
# Personal information keywords
# ---------------------------------------------------------------------------


def update_client_personal_info(
    client: Client_QWIM,
    first_name: str,
    last_name: str,
    current_age: int | str,
    retirement_age: int | str,
    risk_tolerance: int | str,
) -> bool:
    """Call client.update_personal_info with the supplied values.

    Arguments:
    - client          -- Client_QWIM instance
    - first_name      -- updated first name
    - last_name       -- updated last name
    - current_age     -- current age (int or numeric string)
    - retirement_age  -- planned retirement age
    - risk_tolerance  -- integer 1-10

    Returns: True on success, False on failure
    """
    _require_imports()
    data: dict = {
        "First Name": first_name,
        "Last Name": last_name,
        "Current Age": int(current_age),
        "Retirement Age": int(retirement_age),
        "Risk Tolerance": int(risk_tolerance),
        "Income Start Age": int(retirement_age),
    }
    return client.update_personal_info(data)


def update_personal_info_should_return_true(
    client: Client_QWIM,
    first_name: str = "Jane",
    last_name: str = "Doe",
    current_age: int | str = 45,
    retirement_age: int | str = 65,
    risk_tolerance: int | str = 5,
) -> None:
    """Assert that update_personal_info returns True.

    Arguments: same as update_client_personal_info (with defaults).
    """
    _require_imports()
    result = update_client_personal_info(
        client, first_name, last_name, current_age, retirement_age, risk_tolerance
    )
    if not result:
        raise AssertionError("update_personal_info returned False; expected True")


def get_client_current_age(client: Client_QWIM) -> int:
    """Return client.get_current_age().

    Arguments:
    - client -- Client_QWIM instance

    Returns: int current age
    """
    _require_imports()
    return client.get_current_age()


def get_client_retirement_age(client: Client_QWIM) -> int:
    """Return client.get_retirement_age().

    Arguments:
    - client -- Client_QWIM instance

    Returns: int retirement age
    """
    _require_imports()
    return client.get_retirement_age()


def get_client_risk_tolerance(client: Client_QWIM) -> int:
    """Return client.get_risk_tolerance().

    Arguments:
    - client -- Client_QWIM instance

    Returns: int risk tolerance
    """
    _require_imports()
    return client.get_risk_tolerance()


def client_current_age_should_equal(client: Client_QWIM, expected_age: int | str) -> None:
    """Assert that get_current_age() equals expected_age.

    Arguments:
    - client       -- Client_QWIM instance
    - expected_age -- expected age integer
    """
    _require_imports()
    actual = client.get_current_age()
    if actual != int(expected_age):
        raise AssertionError(
            f"Expected current age {expected_age} but got {actual}"
        )


def client_retirement_age_should_equal(client: Client_QWIM, expected: int | str) -> None:
    """Assert that get_retirement_age() equals expected.

    Arguments:
    - client   -- Client_QWIM instance
    - expected -- expected retirement age
    """
    _require_imports()
    actual = client.get_retirement_age()
    if actual != int(expected):
        raise AssertionError(
            f"Expected retirement age {expected} but got {actual}"
        )


def client_risk_tolerance_should_equal(client: Client_QWIM, expected: int | str) -> None:
    """Assert that get_risk_tolerance() equals expected.

    Arguments:
    - client   -- Client_QWIM instance
    - expected -- expected risk tolerance
    """
    _require_imports()
    actual = client.get_risk_tolerance()
    if actual != int(expected):
        raise AssertionError(
            f"Expected risk tolerance {expected} but got {actual}"
        )


def get_personal_info_dataframe(client: Client_QWIM) -> "pl.DataFrame":
    """Return client.get_personal_info() DataFrame.

    Arguments:
    - client -- Client_QWIM instance

    Returns: pl.DataFrame
    """
    _require_imports()
    return client.get_personal_info()


def personal_info_dataframe_should_be_valid(client: Client_QWIM) -> None:
    """Assert that get_personal_info() returns a non-empty DataFrame.

    Arguments:
    - client -- Client_QWIM instance
    """
    _require_imports()
    df = client.get_personal_info()
    if not isinstance(df, pl.DataFrame):
        raise AssertionError(
            f"Expected pl.DataFrame for personal info, got {type(df)}"
        )
    if len(df) == 0:
        raise AssertionError("Personal info DataFrame is empty after update")


# ---------------------------------------------------------------------------
# Assets keywords
# ---------------------------------------------------------------------------


def update_client_assets(
    client: Client_QWIM,
    taxable: float | str = 100000.0,
    tax_deferred: float | str = 200000.0,
    tax_free: float | str = 50000.0,
) -> bool:
    """Call client.update_assets with a single-row asset record.

    Arguments:
    - client        -- Client_QWIM instance
    - taxable       -- taxable asset amount
    - tax_deferred  -- tax-deferred asset amount
    - tax_free      -- tax-free asset amount

    Returns: True on success, False on failure
    """
    _require_imports()
    data: list[dict] = [
        {
            "Taxable Assets": float(taxable),
            "Tax Deferred Assets": float(tax_deferred),
            "Tax Free Assets": float(tax_free),
            "Asset Name": "Test Asset",
            "Asset Class": "Stocks",
        }
    ]
    return client.update_assets(data)


def update_assets_should_return_true(client: Client_QWIM) -> None:
    """Assert that update_assets with sample data returns True.

    Arguments:
    - client -- Client_QWIM instance
    """
    _require_imports()
    result = update_client_assets(client)
    if not result:
        raise AssertionError("update_assets returned False; expected True")


def get_client_total_assets(client: Client_QWIM) -> float:
    """Return client.get_total_assets().

    Arguments:
    - client -- Client_QWIM instance

    Returns: float total assets
    """
    _require_imports()
    return client.get_total_assets()


def client_total_assets_should_equal(
    client: Client_QWIM, expected: float | str
) -> None:
    """Assert that get_total_assets() matches expected value.

    Arguments:
    - client   -- Client_QWIM instance
    - expected -- expected total assets float
    """
    _require_imports()
    actual = client.get_total_assets()
    expected_float = float(expected)
    if abs(actual - expected_float) > 0.01:
        raise AssertionError(
            f"Expected total assets {expected_float} but got {actual}"
        )


def client_total_assets_should_be_positive(client: Client_QWIM) -> None:
    """Assert that get_total_assets() > 0 after update_assets.

    Arguments:
    - client -- Client_QWIM instance
    """
    _require_imports()
    total = client.get_total_assets()
    if total <= 0:
        raise AssertionError(
            f"Expected positive total assets but got {total}"
        )


# ---------------------------------------------------------------------------
# Goals keywords
# ---------------------------------------------------------------------------


def update_client_goals(
    client: Client_QWIM,
    essential_expense: float | str = 60000.0,
    important_expense: float | str = 20000.0,
    aspirational_expense: float | str = 10000.0,
) -> bool:
    """Call client.update_goals with sample expense data.

    Arguments:
    - client               -- Client_QWIM instance
    - essential_expense    -- essential annual expense
    - important_expense    -- important annual expense
    - aspirational_expense -- aspirational annual expense

    Returns: True on success, False on failure
    """
    _require_imports()
    data: list[dict] = [
        {
            "Essential Annual Expense": float(essential_expense),
            "Important Annual Expense": float(important_expense),
            "Aspirational Annual Expense": float(aspirational_expense),
            "Essential Annual Expense is Inflation Indexed": True,
            "Important Annual Expense is Inflation Indexed": False,
            "Aspirational Annual Expense is Inflation Indexed": False,
        }
    ]
    return client.update_goals(data)


def update_goals_should_return_true(client: Client_QWIM) -> None:
    """Assert that update_goals with sample data returns True.

    Arguments:
    - client -- Client_QWIM instance
    """
    _require_imports()
    result = update_client_goals(client)
    if not result:
        raise AssertionError("update_goals returned False; expected True")


def get_client_goals_dataframe(client: Client_QWIM) -> "pl.DataFrame":
    """Return client.get_goals() DataFrame.

    Arguments:
    - client -- Client_QWIM instance

    Returns: pl.DataFrame
    """
    _require_imports()
    return client.get_goals()


def client_goals_dataframe_should_be_valid(client: Client_QWIM) -> None:
    """Assert that get_goals() returns a non-empty DataFrame.

    Arguments:
    - client -- Client_QWIM instance
    """
    _require_imports()
    df = client.get_goals()
    if not isinstance(df, pl.DataFrame):
        raise AssertionError(
            f"Expected pl.DataFrame for goals, got {type(df)}"
        )
    if len(df) == 0:
        raise AssertionError("Goals DataFrame is empty after update")


# ---------------------------------------------------------------------------
# Income keywords
# ---------------------------------------------------------------------------


def update_client_income(
    client: Client_QWIM,
    social_security: float | str = 24000.0,
    pension: float | str = 12000.0,
) -> bool:
    """Call client.update_income with sample income data.

    Arguments:
    - client         -- Client_QWIM instance
    - social_security -- annual social security income
    - pension        -- annual pension income

    Returns: True on success, False on failure
    """
    _require_imports()
    data: list[dict] = [
        {
            "Annual Social Security": float(social_security),
            "Annual Income from Pension": float(pension),
            "Annual Income from Existing Annuity": 0.0,
            "Annual Income from Other Sources": 0.0,
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
    return client.update_income(data)


def update_income_should_return_true(client: Client_QWIM) -> None:
    """Assert that update_income with sample data returns True.

    Arguments:
    - client -- Client_QWIM instance
    """
    _require_imports()
    result = update_client_income(client)
    if not result:
        raise AssertionError("update_income returned False; expected True")
