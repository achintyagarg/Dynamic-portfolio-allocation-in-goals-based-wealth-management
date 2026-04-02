"""Utility functions for client tab operations.

Provides validation, data processing, and formatting functions specifically
for client-related dashboard operations including personal info, assets, and goals.
"""

from __future__ import annotations

import contextlib
import locale

from typing import Any

from shiny import reactive


# Set locale for currency formatting to ensure consistent display
# across different system configurations
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except locale.Error:
    # Fallback for systems without UTF-8 support
    with contextlib.suppress(locale.Error):
        # Use system default if US locale unavailable
        locale.setlocale(locale.LC_ALL, "en_US")

# Import data access utilities
try:
    from src.dashboard.shiny_utils.reactives_shiny import get_value_from_reactives_shiny

    DATA_UTILS_AVAILABLE = True
    print("✅ reactives_shiny utilities imported successfully in utils_tab_clients")
except ImportError as import_error:
    print(f"⚠️ reactives_shiny utilities import failed in utils_tab_clients: {import_error}")
    DATA_UTILS_AVAILABLE = False

    # Fallback function for defensive programming
    def get_value_from_reactives_shiny(
        reactives_shiny: dict,
        key_name: str,
        key_category: str,
    ) -> None:
        """
        Fallback function when reactives_shiny utilities are not available.

        Args:
            reactives_shiny (dict): Reactive values structure
            key_name (str): Key name to retrieve
            key_category (str): Key category

        Returns
        -------
            None: Fallback value
        """
        print(f"⚠️ Fallback: Unable to retrieve {key_name} from {key_category}")
        return


def format_currency_display(amount_value: Any) -> str | None:
    """Format numeric amount as currency with comma separators.

    Converts a numeric value to a formatted currency string following
    the pattern $X,XXX,XXX for easy readability in financial contexts.

    Args:
        amount_value (float | int | None): Numeric amount to format.
            Can be None, which will return "$0".

    Returns
    -------
        str: Formatted currency string in the format "$X,XXX,XXX".

    Raises
    ------
        ValueError: If amount_value cannot be converted to a valid number.

    Examples
    --------
        >>> format_currency_display(1234567)
        '$1,234,567'
        >>> format_currency_display(0)
        '$0'
        >>> format_currency_display(None)
        '$0'

    Note:
        This function uses Python's built-in formatting with comma separators
        and does not include decimal places for whole dollar amounts.
    """
    if amount_value is None:
        return "$0"

    try:
        amount_value = float(amount_value)
        return f"${amount_value:,.0f}"
    except (ValueError, TypeError):
        raise ValueError(f"Invalid amount for currency formatting: {amount_value}")


def validate_financial_amount(amount_value: Any) -> Any:
    """Validate that financial amount meets business requirements.

    Ensures that financial amounts are non-negative numbers,
    following business rules that prevent negative financial goals.

    Args:
        amount_value (float | int | None): Amount to validate.
            None values are converted to 0.0.

    Returns
    -------
        float: Validated amount as a float value.

    Raises
    ------
        ValueError: If amount is negative or cannot be converted to a number.

    Examples
    --------
        >>> validate_financial_amount(1000)
        1000.0
        >>> validate_financial_amount(None)
        0.0
        >>> validate_financial_amount(-500)
        ValueError: Financial amount cannot be negative

    Note:
        This validation implements the business rule that financial amounts
        must be zero or positive values only.
    """
    if amount_value is None:
        return 0.0

    try:
        amount_value = float(amount_value)
        if amount_value < 0:
            raise ValueError("Financial amount cannot be negative")
        return amount_value
    except (ValueError, TypeError) as exc_error:
        if "negative" in str(exc_error):
            raise exc_error
        raise ValueError(f"Invalid financial amount: {amount_value}")


def validate_age_range(
    age_value: int | str | None,
    minimum_age: int | None = None,
    maximum_age: int | None = None,
    age_type_description: str = "Age",
    *,
    min_age: int | None = None,
    max_age: int | None = None,
) -> int:
    """Validate that age value falls within specified range.

    Supports two calling conventions for the range parameters:
    - ``minimum_age`` / ``maximum_age`` (original, used by unit tests)
    - ``min_age`` / ``max_age`` (aliases, used by integration tests)

    Args:
        age_value: Age value to validate
        minimum_age: Minimum allowed age (or use ``min_age``)
        maximum_age: Maximum allowed age (or use ``max_age``)
        age_type_description: Description of age type for error messages
        min_age: Alias for minimum_age (keyword-only)
        max_age: Alias for maximum_age (keyword-only)

    Returns
    -------
        int: Validated age value

    Raises
    ------
        ValueError: If age is outside valid range or invalid
    """
    # Resolve parameter name aliases
    eff_min = minimum_age if minimum_age is not None else min_age
    eff_max = maximum_age if maximum_age is not None else max_age

    if eff_min is None:
        raise ValueError("min_age or minimum_age must be provided")
    if eff_max is None:
        raise ValueError("max_age or maximum_age must be provided")

    if age_value is None:
        raise ValueError(f"{age_type_description} cannot be empty")

    try:
        age_value = int(age_value)
        if age_value < eff_min or age_value > eff_max:
            raise ValueError(
                f"{age_type_description} must be between {eff_min} and {eff_max}",
            )
        return age_value
    except (ValueError, TypeError) as exc_error:
        if "between" in str(exc_error):
            raise exc_error
        raise ValueError(f"Invalid {age_type_description}: {age_value}")


def validate_string_input(
    input_value: str | None,
    field_description: str = "Field",
) -> str:
    """Validate that string input is not empty and properly formatted.

    Args:
        input_value: String value to validate
        field_description: Description of field for error messages

    Returns
    -------
        str: Validated and trimmed string value

    Raises
    ------
        ValueError: If input is empty or invalid
    """
    if input_value is None or str(input_value).strip() == "":
        raise ValueError(f"{field_description} cannot be empty")

    return str(input_value).strip()


def get_investor_data_from_dashboard(
    reactives_shiny: dict,
) -> tuple[dict, dict, dict, dict]:
    """
    Retrieve investor personal information, assets, goals, and income data from dashboard inputs.

    This function extracts comprehensive investor data from the Shiny dashboard reactive structure,
    including personal demographics, asset allocations, financial goals, and income projections
    for both primary and partner investors. The function implements defensive programming with
    comprehensive error handling and validation.

    Args:
        reactives_shiny (dict): Reactive values structure containing dashboard inputs.
            Expected to contain User_Inputs_Shiny category with hierarchical key structure
            following QWIM naming conventions.

    Returns
    -------
        Tuple[dict, dict, dict, dict]: Four dictionaries containing:
            - personal_info_data: Personal demographics for primary and partner investors
            - assets_data: Asset allocations by tax treatment category
            - goals_data: Financial goals by priority level
            - income_data: Income projections by source category

    Raises
    ------
        ValueError: If reactives_shiny structure is invalid or missing required data
        RuntimeError: If data utilities are not available or configuration issues occur
        KeyError: If required User_Inputs_Shiny category is missing
        Exception: For unexpected errors during data retrieval process

    Examples
    --------
        >>> personal_info, assets, goals, income = get_investor_data_from_dashboard(reactives_shiny)
        >>> print(f"Primary investor: {personal_info['primary']['name']}")
        >>> print(f"Total assets: ${assets['combined']['total']:,.0f}")

    Note:
        Function includes comprehensive logging for debugging and monitoring purposes.
        All financial values are validated and converted to float with defensive programming.
        All errors are explicitly raised rather than using fallback data.
    """
    try:
        print("🔄 Starting comprehensive investor data retrieval from dashboard")

        # Input validation with early returns
        if not reactives_shiny or not isinstance(reactives_shiny, dict):
            Error_Message = "Invalid reactives_shiny structure - must be a non-empty dictionary"
            print(f"❌ Error: {Error_Message}")
            raise ValueError(Error_Message)

        # Configuration validation - check if data utilities are available
        if not DATA_UTILS_AVAILABLE:
            Error_Message = "Data utilities not available - reactives_shiny module failed to import"
            print(f"❌ Error: {Error_Message}")
            raise RuntimeError(Error_Message)

        # Business logic validation - check if User_Inputs_Shiny category exists
        if "User_Inputs_Shiny" not in reactives_shiny:
            Error_Message = f"User_Inputs_Shiny category not found in reactives_shiny. Available categories: {list(reactives_shiny.keys())}"
            print(f"❌ Error: {Error_Message}")
            raise KeyError(Error_Message)

        print("📊 Retrieving primary investor personal information...")
        # Primary Investor Personal Information
        primary_personal_info = get_investor_primary_personal_info(reactives_shiny)

        print("📊 Retrieving partner investor personal information...")
        # Partner Investor Personal Information
        partner_personal_info = get_investor_partner_personal_info(reactives_shiny)

        print("💰 Retrieving primary investor assets...")
        # Primary Investor Assets
        primary_assets = get_investor_primary_assets(reactives_shiny)

        print("💰 Retrieving partner investor assets...")
        # Partner Investor Assets
        partner_assets = get_investor_partner_assets(reactives_shiny)

        print("🎯 Retrieving primary investor goals...")
        # Primary Investor Goals
        primary_goals = get_investor_primary_goals(reactives_shiny)

        print("🎯 Retrieving partner investor goals...")
        # Partner Investor Goals
        partner_goals = get_investor_partner_goals(reactives_shiny)

        print("💵 Retrieving primary investor income...")
        # Primary Investor Income
        primary_income = get_investor_primary_income(reactives_shiny)

        print("💵 Retrieving partner investor income...")
        # Partner Investor Income
        partner_income = get_investor_partner_income(reactives_shiny)

        print("🧮 Calculating totals and combined values...")
        # Calculate totals with defensive programming
        _calculate_individual_totals(
            primary_assets,
            partner_assets,
            primary_goals,
            partner_goals,
            primary_income,
            partner_income,
        )

        # Calculate combined totals
        combined_assets = _calculate_combined_assets(primary_assets, partner_assets)
        combined_goals = _calculate_combined_goals(primary_goals, partner_goals)
        combined_income = _calculate_combined_income(primary_income, partner_income)

        # Structure the data for return
        personal_info_data = {
            "primary": primary_personal_info,
            "partner": partner_personal_info,
        }

        assets_data = {
            "primary": primary_assets,
            "partner": partner_assets,
            "combined": combined_assets,
        }

        goals_data = {
            "primary": primary_goals,
            "partner": partner_goals,
            "combined": combined_goals,
        }

        income_data = {
            "primary": primary_income,
            "partner": partner_income,
            "combined": combined_income,
        }

        print("✅ Successfully retrieved comprehensive investor data from dashboard")
        print(f"   Primary investor: {primary_personal_info['name']}")
        print(f"   Partner investor: {partner_personal_info['name']}")
        print(f"   Primary total assets: ${primary_assets['total']:,.0f}")
        print(f"   Partner total assets: ${partner_assets['total']:,.0f}")
        print(f"   Combined total assets: ${combined_assets['total']:,.0f}")
        print(f"   Primary total goals: ${primary_goals['total']:,.0f}")
        print(f"   Partner total goals: ${partner_goals['total']:,.0f}")
        print(f"   Combined total goals: ${combined_goals['total']:,.0f}")
        print(f"   Primary total income: ${primary_income['total']:,.0f}")
        print(f"   Partner total income: ${partner_income['total']:,.0f}")
        print(f"   Combined total income: ${combined_income['total']:,.0f}")

        return personal_info_data, assets_data, goals_data, income_data

    except (ValueError, RuntimeError, KeyError) as known_error:
        # Re-raise known errors with context
        print(f"❌ Known error during investor data retrieval: {known_error}")
        raise known_error

    except Exception as data_error:
        Error_Message = (
            f"Unexpected error occurred during comprehensive investor data retrieval: {data_error}"
        )
        print(f"❌ Error: {Error_Message}")
        print(f"   Error type: {type(data_error).__name__}")
        import traceback

        print(f"   Traceback: {traceback.format_exc()}")
        raise Exception(Error_Message) from data_error


def get_investor_primary_personal_info(reactives_shiny: dict) -> dict:
    """
    Retrieve primary client personal information from dashboard inputs.

    Args:
        reactives_shiny (dict): Reactive values structure

    Returns
    -------
        dict: Primary client personal information
    """
    return {
        "name": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Name",
            key_category="User_Inputs_Shiny",
        )
        or "Primary Client",
        "age_current": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Current",
            key_category="User_Inputs_Shiny",
        )
        or 35,
        "age_retirement": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Retirement",
            key_category="User_Inputs_Shiny",
        )
        or 65,
        "age_income_starting": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Income_Starting",
            key_category="User_Inputs_Shiny",
        )
        or 67,
        "status_marital": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Status_Marital",
            key_category="User_Inputs_Shiny",
        )
        or "Not Specified",
        "gender": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Gender",
            key_category="User_Inputs_Shiny",
        )
        or "Not Specified",
        "tolerance_risk": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Tolerance_Risk",
            key_category="User_Inputs_Shiny",
        )
        or "Moderate",
        "state": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_State",
            key_category="User_Inputs_Shiny",
        )
        or "Not Specified",
        "code_zip": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Code_Zip",
            key_category="User_Inputs_Shiny",
        )
        or "00000",
    }


def get_investor_partner_personal_info(reactives_shiny: dict) -> dict:
    """
    Retrieve partner client personal information from dashboard inputs.

    Args:
        reactives_shiny (dict): Reactive values structure

    Returns
    -------
        dict: Partner client personal information
    """
    return {
        "name": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Name",
            key_category="User_Inputs_Shiny",
        )
        or "Partner Client",
        "age_current": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Current",
            key_category="User_Inputs_Shiny",
        )
        or 33,
        "age_retirement": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Retirement",
            key_category="User_Inputs_Shiny",
        )
        or 65,
        "age_income_starting": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Income_Starting",
            key_category="User_Inputs_Shiny",
        )
        or 67,
        "status_marital": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Status_Marital",
            key_category="User_Inputs_Shiny",
        )
        or "Not Specified",
        "gender": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Gender",
            key_category="User_Inputs_Shiny",
        )
        or "Not Specified",
        "tolerance_risk": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Tolerance_Risk",
            key_category="User_Inputs_Shiny",
        )
        or "Moderate",
        "state": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_State",
            key_category="User_Inputs_Shiny",
        )
        or "Not Specified",
        "code_zip": get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Code_Zip",
            key_category="User_Inputs_Shiny",
        )
        or "00000",
    }


def get_investor_primary_assets(reactives_shiny: dict) -> dict:
    """
    Retrieve primary investor assets from dashboard inputs.

    Args:
        reactives_shiny (dict): Reactive values structure

    Returns
    -------
        dict: Primary investor assets by category
    """
    return {
        "taxable": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Primary_Assets_Taxable",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "tax_deferred": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Primary_Assets_Tax_Deferred",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "tax_free": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Primary_Assets_Tax_Free",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
    }


def get_investor_partner_assets(reactives_shiny: dict) -> dict:
    """
    Retrieve partner investor assets from dashboard inputs.

    Args:
        reactives_shiny (dict): Reactive values structure

    Returns
    -------
        dict: Partner investor assets by category
    """
    return {
        "taxable": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Partner_Assets_Taxable",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "tax_deferred": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Partner_Assets_Tax_Deferred",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "tax_free": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Partner_Assets_Tax_Free",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
    }


def get_investor_primary_goals(reactives_shiny: dict) -> dict:
    """
    Retrieve primary investor goals from dashboard inputs.

    Args:
        reactives_shiny (dict): Reactive values structure

    Returns
    -------
        dict: Primary investor goals by priority level
    """
    return {
        "essential": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Essential",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "important": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Important",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "aspirational": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Aspirational",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
    }


def get_investor_partner_goals(reactives_shiny: dict) -> dict:
    """
    Retrieve partner investor goals from dashboard inputs.

    Args:
        reactives_shiny (dict): Reactive values structure

    Returns
    -------
        dict: Partner investor goals by priority level
    """
    return {
        "essential": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Essential",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "important": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Important",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "aspirational": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Aspirational",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
    }


def get_investor_primary_income(reactives_shiny: dict) -> dict:
    """
    Retrieve primary investor income from dashboard inputs.

    Args:
        reactives_shiny (dict): Reactive values structure

    Returns
    -------
        dict: Primary investor income by source category
    """
    return {
        "social_security": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Social_Security",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "pension": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Pension",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "annuity_existing": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Annuity_Existing",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "other": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Other",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
    }


def get_investor_partner_income(reactives_shiny: dict) -> dict:
    """
    Retrieve partner investor income from dashboard inputs.

    Args:
        reactives_shiny (dict): Reactive values structure

    Returns
    -------
        dict: Partner investor income by source category
    """
    return {
        "social_security": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Social_Security",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "pension": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Pension",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "annuity_existing": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Annuity_Existing",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
        "other": float(
            get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Other",
                key_category="User_Inputs_Shiny",
            )
            or 0.0,
        ),
    }


# Observer functions for reactive behavior in the dashboard
@reactive.effect
def observer_get_investor_primary_personal_info() -> None:
    """Observer for primary investor personal information changes."""


@reactive.effect
def observer_get_investor_partner_personal_info() -> None:
    """Observer for partner investor personal information changes."""


@reactive.effect
def observer_get_investor_primary_assets() -> None:
    """Observer for primary investor assets changes."""


@reactive.effect
def observer_get_investor_partner_assets() -> None:
    """Observer for partner investor assets changes."""


@reactive.effect
def observer_get_investor_primary_goals() -> None:
    """Observer for primary investor goals changes."""


@reactive.effect
def observer_get_investor_partner_goals() -> None:
    """Observer for partner investor goals changes."""


@reactive.effect
def observer_get_investor_primary_income() -> None:
    """Observer for primary investor income changes."""


@reactive.effect
def observer_get_investor_partner_income() -> None:
    """Observer for partner investor income changes."""


def _calculate_individual_totals(
    primary_assets: dict,
    partner_assets: dict,
    primary_goals: dict,
    partner_goals: dict,
    primary_income: dict,
    partner_income: dict,
) -> None:
    """
    Calculate individual totals for assets, goals, and income.

    Args:
        primary_assets (dict): Primary investor assets
        partner_assets (dict): Partner investor assets
        primary_goals (dict): Primary investor goals
        partner_goals (dict): Partner investor goals
        primary_income (dict): Primary investor income
        partner_income (dict): Partner investor income
    """
    try:
        # Asset totals
        primary_assets["total"] = (
            primary_assets["taxable"] + primary_assets["tax_deferred"] + primary_assets["tax_free"]
        )
        partner_assets["total"] = (
            partner_assets["taxable"] + partner_assets["tax_deferred"] + partner_assets["tax_free"]
        )

        # Goals totals
        primary_goals["total"] = (
            primary_goals["essential"] + primary_goals["important"] + primary_goals["aspirational"]
        )
        partner_goals["total"] = (
            partner_goals["essential"] + partner_goals["important"] + partner_goals["aspirational"]
        )

        # Income totals
        primary_income["total"] = (
            primary_income["social_security"]
            + primary_income["pension"]
            + primary_income["annuity_existing"]
            + primary_income["other"]
        )
        partner_income["total"] = (
            partner_income["social_security"]
            + partner_income["pension"]
            + partner_income["annuity_existing"]
            + partner_income["other"]
        )

    except (TypeError, ValueError) as calculation_error:
        print(f"⚠️ Error calculating individual totals: {calculation_error}")
        # Set defaults for defensive programming
        primary_assets["total"] = 0.0
        partner_assets["total"] = 0.0
        primary_goals["total"] = 0.0
        partner_goals["total"] = 0.0
        primary_income["total"] = 0.0
        partner_income["total"] = 0.0


def _calculate_combined_assets(
    primary_assets: dict,
    partner_assets: dict,
) -> dict:
    """
    Calculate combined assets for both investors.

    Args:
        primary_assets (dict): Primary investor assets
        partner_assets (dict): Partner investor assets

    Returns
    -------
        dict: Combined assets by category
    """
    try:
        combined_assets = {
            "taxable": primary_assets["taxable"] + partner_assets["taxable"],
            "tax_deferred": primary_assets["tax_deferred"] + partner_assets["tax_deferred"],
            "tax_free": primary_assets["tax_free"] + partner_assets["tax_free"],
        }
        combined_assets["total"] = (
            combined_assets["taxable"]
            + combined_assets["tax_deferred"]
            + combined_assets["tax_free"]
        )
        return combined_assets
    except (TypeError, ValueError) as calculation_error:
        print(f"⚠️ Error calculating combined assets: {calculation_error}")
        return {"taxable": 0.0, "tax_deferred": 0.0, "tax_free": 0.0, "total": 0.0}


def _calculate_combined_goals(
    primary_goals: dict,
    partner_goals: dict,
) -> dict:
    """
    Calculate combined goals for both investors.

    Args:
        primary_goals (dict): Primary investor goals
        partner_goals (dict): Partner investor goals

    Returns
    -------
        dict: Combined goals by priority level
    """
    try:
        combined_goals = {
            "essential": primary_goals["essential"] + partner_goals["essential"],
            "important": primary_goals["important"] + partner_goals["important"],
            "aspirational": primary_goals["aspirational"] + partner_goals["aspirational"],
        }
        combined_goals["total"] = (
            combined_goals["essential"]
            + combined_goals["important"]
            + combined_goals["aspirational"]
        )
        return combined_goals
    except (TypeError, ValueError) as calculation_error:
        print(f"⚠️ Error calculating combined goals: {calculation_error}")
        return {"essential": 0.0, "important": 0.0, "aspirational": 0.0, "total": 0.0}


def _calculate_combined_income(
    primary_income: dict,
    partner_income: dict,
) -> dict:
    """
    Calculate combined income for both investors.

    Args:
        primary_income (dict): Primary investor income
        partner_income (dict): Partner investor income

    Returns
    -------
        dict: Combined income by source category
    """
    try:
        combined_income = {
            "social_security": primary_income["social_security"]
            + partner_income["social_security"],
            "pension": primary_income["pension"] + partner_income["pension"],
            "annuity_existing": primary_income["annuity_existing"]
            + partner_income["annuity_existing"],
            "other": primary_income["other"] + partner_income["other"],
        }
        combined_income["total"] = (
            combined_income["social_security"]
            + combined_income["pension"]
            + combined_income["annuity_existing"]
            + combined_income["other"]
        )
        return combined_income
    except (TypeError, ValueError) as calculation_error:
        print(f"⚠️ Error calculating combined income: {calculation_error}")
        return {
            "social_security": 0.0,
            "pension": 0.0,
            "annuity_existing": 0.0,
            "other": 0.0,
            "total": 0.0,
        }
