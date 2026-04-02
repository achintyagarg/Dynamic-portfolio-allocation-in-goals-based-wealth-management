"""Utility functions for reporting.

Provides utilities for reporting (such as PDF reports generated in Shiny dashboard)
and for saving and retrieving client input data within the Shiny reactives data
structure (``reactives_shiny["Data_Clients"]``), as well as subtab results data
within ``reactives_shiny["Data_Results"]``.

The ``Data_Clients`` category in the reactives_shiny structure has the following
nested layout::

    {
        "Single_Or_Couple": reactive.Value("Single" | "Couple"),
        "Client_Primary": {
            "Personal_Info": reactive.Value(pl.DataFrame | None),
            "Assets": reactive.Value(pl.DataFrame | None),
            "Goals": reactive.Value(pl.DataFrame | None),
            "Income": reactive.Value(pl.DataFrame | None),
        },
        "Client_Partner": {
            "Personal_Info": reactive.Value(pl.DataFrame | None),
            "Assets": reactive.Value(pl.DataFrame | None),
            "Goals": reactive.Value(pl.DataFrame | None),
            "Income": reactive.Value(pl.DataFrame | None),
        },
        "Clients_Combined": {
            "Assets": reactive.Value(pl.DataFrame | None),
            "Goals": reactive.Value(pl.DataFrame | None),
            "Income": reactive.Value(pl.DataFrame | None),
        },
    }

The ``Data_Results`` category in the reactives_shiny structure has the following
flat layout (one reactive.Value per subtab/direction combination)::

    {
        "Portfolio_Analysis_Inputs": reactive.Value(None),
        "Portfolio_Analysis_Outputs": reactive.Value(None),
        "Portfolio_Comparison_Inputs": reactive.Value(None),
        "Portfolio_Comparison_Outputs": reactive.Value(None),
        "Weights_Analysis_Inputs": reactive.Value(None),
        "Weights_Analysis_Outputs": reactive.Value(None),
        "Portfolio_Optimization_Skfolio_Inputs": reactive.Value(None),
        "Portfolio_Optimization_Skfolio_Outputs": reactive.Value(None),
        "Portfolio_Simulation_Inputs": reactive.Value(None),
        "Portfolio_Simulation_Outputs": reactive.Value(None),
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)

#: Valid client levels within the Data_Clients structure
VALID_CLIENT_LEVELS: frozenset[str] = frozenset(
    ["Client_Primary", "Client_Partner", "Clients_Combined"],
)

#: Valid data categories for a single client (primary or partner)
VALID_CLIENT_DATA_CATEGORIES_PER_CLIENT: frozenset[str] = frozenset(
    ["Personal_Info", "Assets", "Goals", "Income"],
)

#: Valid data categories for combined client data (no Personal_Info)
VALID_CLIENT_DATA_CATEGORIES_COMBINED: frozenset[str] = frozenset(
    ["Assets", "Goals", "Income"],
)

#: Allowed values for the Single_Or_Couple indicator
VALID_SINGLE_OR_COUPLE_VALUES: frozenset[str] = frozenset(["Single", "Couple"])


def validate_data_clients_in_reactives(reactives_shiny: Any) -> tuple[bool, str]:
    """Validate that the Data_Clients category exists and has the required structure.

    Checks that ``reactives_shiny["Data_Clients"]`` is present, is a dictionary,
    contains ``Single_Or_Couple`` and the three sub-level keys (``Client_Primary``,
    ``Client_Partner``, ``Clients_Combined``), each of which is itself a dictionary.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.

    Returns
    -------
        tuple[bool, str]: ``(True, "")`` when the structure is valid, or
        ``(False, error_message)`` with a descriptive error message otherwise.
    """
    # Input validation with early returns
    if reactives_shiny is None:
        return False, "reactives_shiny dictionary cannot be None"

    if not isinstance(reactives_shiny, dict):
        return (
            False,
            f"reactives_shiny must be a dictionary, got {type(reactives_shiny).__name__}",
        )

    # Configuration validation - Data_Clients category must exist
    if "Data_Clients" not in reactives_shiny:
        available_categories = list(reactives_shiny.keys())
        return (
            False,
            f"'Data_Clients' category not found in reactives_shiny. "
            f"Available categories: {available_categories}",
        )

    data_clients = reactives_shiny["Data_Clients"]

    if not isinstance(data_clients, dict):
        return (
            False,
            f"'Data_Clients' must be a dictionary, got {type(data_clients).__name__}",
        )

    # Business logic validation - Single_Or_Couple must exist
    if "Single_Or_Couple" not in data_clients:
        available_keys = list(data_clients.keys())
        return (
            False,
            f"'Single_Or_Couple' key not found in Data_Clients. Available keys: {available_keys}",
        )

    # Business logic validation - sub-level dicts must exist and be dicts
    required_sub_levels = ["Client_Primary", "Client_Partner", "Clients_Combined"]
    for sub_level_name in required_sub_levels:
        if sub_level_name not in data_clients:
            available_keys = list(data_clients.keys())
            return (
                False,
                f"Sub-level '{sub_level_name}' not found in Data_Clients. "
                f"Available keys: {available_keys}",
            )
        if not isinstance(data_clients[sub_level_name], dict):
            return (
                False,
                f"Sub-level '{sub_level_name}' in Data_Clients must be a dictionary, "
                f"got {type(data_clients[sub_level_name]).__name__}",
            )

    return True, ""


def validate_client_level_name(client_level: Any) -> tuple[bool, str]:
    """Validate that the provided client level name is one of the allowed sub-levels.

    The allowed sub-levels mirror the three keys inside ``Data_Clients``:
    ``"Client_Primary"``, ``"Client_Partner"``, and ``"Clients_Combined"``.

    Args:
        client_level: The client level name to validate.

    Returns
    -------
        tuple[bool, str]: ``(True, "")`` when valid, or
        ``(False, error_message)`` when invalid.
    """
    # Input validation with early returns
    if client_level is None:
        return False, "client_level cannot be None"

    if not isinstance(client_level, str):
        return False, f"client_level must be a string, got {type(client_level).__name__}"

    if len(client_level.strip()) == 0:
        return False, "client_level cannot be an empty string"

    # Business logic validation
    if client_level not in VALID_CLIENT_LEVELS:
        return (
            False,
            f"Invalid client_level '{client_level}'. Valid levels: {sorted(VALID_CLIENT_LEVELS)}",
        )

    return True, ""


def validate_client_data_category_name(
    data_category: Any,
    client_level: Any = None,
) -> tuple[bool, str]:
    """Validate that the data category is allowed for the given client level.

    ``"Personal_Info"`` is only valid for ``"Client_Primary"`` and
    ``"Client_Partner"``, not for ``"Clients_Combined"``.

    Args:
        data_category: The data category name to validate (e.g. ``"Personal_Info"``,
            ``"Assets"``, ``"Goals"``, ``"Income"``).
        client_level: Optional client level used to apply level-specific
            category restrictions.  When ``None`` the broader set of all
            categories is accepted.

    Returns
    -------
        tuple[bool, str]: ``(True, "")`` when valid, or
        ``(False, error_message)`` when invalid.
    """
    # Input validation with early returns
    if data_category is None:
        return False, "data_category cannot be None"

    if not isinstance(data_category, str):
        return False, f"data_category must be a string, got {type(data_category).__name__}"

    if len(data_category.strip()) == 0:
        return False, "data_category cannot be an empty string"

    # Business logic validation - apply level-specific restrictions
    if client_level == "Clients_Combined":
        if data_category not in VALID_CLIENT_DATA_CATEGORIES_COMBINED:
            return (
                False,
                f"Invalid data_category '{data_category}' for 'Clients_Combined'. "
                f"Valid categories: {sorted(VALID_CLIENT_DATA_CATEGORIES_COMBINED)}",
            )
    else:
        # For Client_Primary, Client_Partner, or unknown level use the full set
        all_valid = VALID_CLIENT_DATA_CATEGORIES_PER_CLIENT | VALID_CLIENT_DATA_CATEGORIES_COMBINED
        if data_category not in all_valid:
            return (
                False,
                f"Invalid data_category '{data_category}'. Valid categories: {sorted(all_valid)}",
            )

    return True, ""


def get_single_or_couple_from_reactives(reactives_shiny: Any) -> str:
    """Retrieve the Single_Or_Couple indicator from the reactives data structure.

    Reads ``reactives_shiny["Data_Clients"]["Single_Or_Couple"]`` and returns its
    current value (either ``"Single"`` or ``"Couple"``).

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.

    Returns
    -------
        str: ``"Single"`` when the household has only a primary client, or
        ``"Couple"`` when it includes both a primary and a partner client.

    Raises
    ------
        ValueError: If the reactives structure is invalid or the reactive value is missing.
        RuntimeError: If the reactive value cannot be retrieved.
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_clients_in_reactives(reactives_shiny)
    if not validation_result:
        error_message = f"Data_Clients structure validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    # Business logic validation - retrieve Single_Or_Couple reactive
    reactive_single_or_couple = reactives_shiny["Data_Clients"]["Single_Or_Couple"]

    if reactive_single_or_couple is None:
        error_message = (
            "Reactive variable 'Single_Or_Couple' in Data_Clients is None - "
            "it was not properly initialized"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    if not hasattr(reactive_single_or_couple, "get"):
        error_message = (
            f"Reactive variable 'Single_Or_Couple' in Data_Clients does not have a 'get' method. "
            f"Object type: {type(reactive_single_or_couple).__name__}"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    # Only use try-except for unpredictable reactive access
    try:
        value = reactive_single_or_couple.get()
        _logger.debug(
            "Retrieved Single_Or_Couple from reactives",
            extra={"single_or_couple": value},
        )
        return value if value is not None else "Single"

    except Exception as exc:
        error_message = f"Unexpected error retrieving 'Single_Or_Couple' from Data_Clients: {exc}"
        _logger.error(error_message)
        raise RuntimeError(error_message) from exc


def update_single_or_couple_in_reactives(
    reactives_shiny: Any,
    single_or_couple: Any,
) -> dict:
    """Update the Single_Or_Couple indicator in the reactives data structure.

    Sets ``reactives_shiny["Data_Clients"]["Single_Or_Couple"]`` to the provided
    value, which must be either ``"Single"`` or ``"Couple"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        single_or_couple: New value to store.  Must be ``"Single"`` or ``"Couple"``.

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If the reactives structure or the provided value is invalid.
        RuntimeError: If the reactive value cannot be updated.
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_clients_in_reactives(reactives_shiny)
    if not validation_result:
        error_message = f"Data_Clients structure validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    if single_or_couple is None:
        error_message = "single_or_couple value cannot be None"
        _logger.error(error_message)
        raise ValueError(error_message)

    if not isinstance(single_or_couple, str):
        error_message = f"single_or_couple must be a string, got {type(single_or_couple).__name__}"
        _logger.error(error_message)
        raise ValueError(error_message)

    # Business logic validation - must be "Single" or "Couple"
    if single_or_couple not in VALID_SINGLE_OR_COUPLE_VALUES:
        error_message = (
            f"Invalid single_or_couple value '{single_or_couple}'. "
            f"Valid values: {sorted(VALID_SINGLE_OR_COUPLE_VALUES)}"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    reactive_single_or_couple = reactives_shiny["Data_Clients"]["Single_Or_Couple"]

    if not hasattr(reactive_single_or_couple, "set"):
        error_message = (
            f"Reactive variable 'Single_Or_Couple' does not have a 'set' method. "
            f"Object type: {type(reactive_single_or_couple).__name__}"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    # Only use try-except for unpredictable reactive access
    try:
        reactive_single_or_couple.set(single_or_couple)
        _logger.debug(
            "Updated Single_Or_Couple in reactives",
            extra={"single_or_couple": single_or_couple},
        )
        return reactives_shiny

    except Exception as exc:
        error_message = (
            f"Unexpected error updating 'Single_Or_Couple' to '{single_or_couple}': {exc}"
        )
        _logger.error(error_message)
        raise RuntimeError(error_message) from exc


def get_client_data_from_reactives(
    reactives_shiny: Any,
    client_level: Any,
    data_category: Any,
) -> pl.DataFrame | None:
    """Retrieve client input data stored at a specific level and category in reactives.

    Accesses ``reactives_shiny["Data_Clients"][client_level][data_category]`` and
    returns the current reactive value (a Polars DataFrame or ``None``).

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        client_level: Sub-level key.  One of ``"Client_Primary"``,
            ``"Client_Partner"``, or ``"Clients_Combined"``.
        data_category: Data category key.  One of ``"Personal_Info"``, ``"Assets"``,
            ``"Goals"``, ``"Income"`` (``"Personal_Info"`` is not valid for
            ``"Clients_Combined"``).

    Returns
    -------
        pl.DataFrame | None: The stored Polars DataFrame, or ``None`` when no data
        has been saved yet for the given level / category combination.

    Raises
    ------
        ValueError: If the reactives structure, client_level, or data_category is invalid.
        KeyError: If the requested key is not found inside the sub-level dictionary.
        RuntimeError: If the reactive value cannot be retrieved.
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_clients_in_reactives(reactives_shiny)
    if not validation_result:
        error_message = f"Data_Clients structure validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    validation_result, validation_message = validate_client_level_name(client_level)
    if not validation_result:
        error_message = f"Client level validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    validation_result, validation_message = validate_client_data_category_name(
        data_category,
        client_level,
    )
    if not validation_result:
        error_message = f"Data category validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    # Configuration validation - sub-level dict must contain requested category
    sub_level_dict = reactives_shiny["Data_Clients"][client_level]

    if data_category not in sub_level_dict:
        available_categories = list(sub_level_dict.keys())
        error_message = (
            f"Data category '{data_category}' not found in Data_Clients['{client_level}']. "
            f"Available categories: {available_categories}"
        )
        _logger.error(error_message)
        raise KeyError(error_message)

    reactive_variable = sub_level_dict[data_category]

    # Business logic validation
    if reactive_variable is None:
        error_message = (
            f"Reactive variable at Data_Clients['{client_level}']['{data_category}'] is None - "
            f"it was not properly initialized"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    if not hasattr(reactive_variable, "get"):
        error_message = (
            f"Reactive variable at Data_Clients['{client_level}']['{data_category}'] "
            f"does not have a 'get' method. "
            f"Object type: {type(reactive_variable).__name__}"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    # Only use try-except for unpredictable reactive access
    try:
        retrieved_value = reactive_variable.get()
        _logger.debug(
            "Retrieved client data from reactives",
            extra={
                "client_level": client_level,
                "data_category": data_category,
                "value_type": type(retrieved_value).__name__,
            },
        )
        return retrieved_value

    except Exception as exc:
        error_message = (
            f"Unexpected error retrieving Data_Clients['{client_level}']['{data_category}']: {exc}"
        )
        _logger.error(error_message)
        raise RuntimeError(error_message) from exc


def save_client_data_to_reactives(
    reactives_shiny: Any,
    client_level: Any,
    data_category: Any,
    data_value: Any,
) -> dict:
    """Save client input data to a specific level and category in the reactives structure.

    Sets ``reactives_shiny["Data_Clients"][client_level][data_category]`` to the
    provided Polars DataFrame (or ``None`` to clear a previously stored value).

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        client_level: Sub-level key.  One of ``"Client_Primary"``,
            ``"Client_Partner"``, or ``"Clients_Combined"``.
        data_category: Data category key.  One of ``"Personal_Info"``, ``"Assets"``,
            ``"Goals"``, ``"Income"`` (``"Personal_Info"`` is not valid for
            ``"Clients_Combined"``).
        data_value: Polars DataFrame to store, or ``None`` to clear the entry.

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If the reactives structure, client_level, data_category, or
            data_value type is invalid.
        KeyError: If the requested key is not found inside the sub-level dictionary.
        RuntimeError: If the reactive value cannot be updated.
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_clients_in_reactives(reactives_shiny)
    if not validation_result:
        error_message = f"Data_Clients structure validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    validation_result, validation_message = validate_client_level_name(client_level)
    if not validation_result:
        error_message = f"Client level validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    validation_result, validation_message = validate_client_data_category_name(
        data_category,
        client_level,
    )
    if not validation_result:
        error_message = f"Data category validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    # Business logic validation - data_value must be a pl.DataFrame or None
    if data_value is not None and not isinstance(data_value, pl.DataFrame):
        error_message = (
            f"data_value must be a Polars DataFrame or None, got {type(data_value).__name__}"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    # Configuration validation - sub-level dict must contain requested category
    sub_level_dict = reactives_shiny["Data_Clients"][client_level]

    if data_category not in sub_level_dict:
        available_categories = list(sub_level_dict.keys())
        error_message = (
            f"Data category '{data_category}' not found in Data_Clients['{client_level}']. "
            f"Available categories: {available_categories}"
        )
        _logger.error(error_message)
        raise KeyError(error_message)

    reactive_variable = sub_level_dict[data_category]

    if reactive_variable is None:
        error_message = (
            f"Reactive variable at Data_Clients['{client_level}']['{data_category}'] is None - "
            f"it was not properly initialized"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    if not hasattr(reactive_variable, "set"):
        error_message = (
            f"Reactive variable at Data_Clients['{client_level}']['{data_category}'] "
            f"does not have a 'set' method. "
            f"Object type: {type(reactive_variable).__name__}"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    # Only use try-except for unpredictable reactive access
    try:
        reactive_variable.set(data_value)
        value_shape = data_value.shape if data_value is not None else None
        _logger.debug(
            "Saved client data to reactives",
            extra={
                "client_level": client_level,
                "data_category": data_category,
                "data_shape": str(value_shape),
            },
        )
        return reactives_shiny

    except Exception as exc:
        error_message = (
            f"Unexpected error saving data to "
            f"Data_Clients['{client_level}']['{data_category}']: {exc}"
        )
        _logger.error(error_message)
        raise RuntimeError(error_message) from exc


def save_client_personal_info_to_reactives(
    reactives_shiny: Any,
    client_level: Any,
    data_personal_info: Any,
) -> dict:
    """Save personal information data for a client to the reactives structure.

    Convenience wrapper around :func:`save_client_data_to_reactives` that fixes
    ``data_category`` to ``"Personal_Info"``.  Valid only for ``"Client_Primary"``
    and ``"Client_Partner"`` (not ``"Clients_Combined"``).

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        client_level: Sub-level key.  Must be ``"Client_Primary"`` or
            ``"Client_Partner"``; ``"Clients_Combined"`` is not accepted.
        data_personal_info: Polars DataFrame containing personal-information fields,
            or ``None`` to clear the entry.

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If client_level is ``"Clients_Combined"`` or any other validation
            fails (see :func:`save_client_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    # Business logic validation - Personal_Info is not valid for Clients_Combined
    if client_level == "Clients_Combined":
        error_message = (
            "'Personal_Info' data category is not valid for 'Clients_Combined'. "
            "Use 'Client_Primary' or 'Client_Partner' instead."
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    return save_client_data_to_reactives(
        reactives_shiny,
        client_level,
        "Personal_Info",
        data_personal_info,
    )


def save_client_assets_to_reactives(
    reactives_shiny: Any,
    client_level: Any,
    data_assets: Any,
) -> dict:
    """Save assets data for a client (or the combined household) to the reactives structure.

    Convenience wrapper around :func:`save_client_data_to_reactives` that fixes
    ``data_category`` to ``"Assets"``.  Valid for all three client levels.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        client_level: Sub-level key.  One of ``"Client_Primary"``,
            ``"Client_Partner"``, or ``"Clients_Combined"``.
        data_assets: Polars DataFrame containing asset information, or ``None`` to
            clear the entry.

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_client_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_client_data_to_reactives(
        reactives_shiny,
        client_level,
        "Assets",
        data_assets,
    )


def save_client_goals_to_reactives(
    reactives_shiny: Any,
    client_level: Any,
    data_goals: Any,
) -> dict:
    """Save goals data for a client (or the combined household) to the reactives structure.

    Convenience wrapper around :func:`save_client_data_to_reactives` that fixes
    ``data_category`` to ``"Goals"``.  Valid for all three client levels.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        client_level: Sub-level key.  One of ``"Client_Primary"``,
            ``"Client_Partner"``, or ``"Clients_Combined"``.
        data_goals: Polars DataFrame containing goal information, or ``None`` to
            clear the entry.

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_client_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_client_data_to_reactives(
        reactives_shiny,
        client_level,
        "Goals",
        data_goals,
    )


def save_client_income_to_reactives(
    reactives_shiny: Any,
    client_level: Any,
    data_income: Any,
) -> dict:
    """Save income data for a client (or the combined household) to the reactives structure.

    Convenience wrapper around :func:`save_client_data_to_reactives` that fixes
    ``data_category`` to ``"Income"``.  Valid for all three client levels.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        client_level: Sub-level key.  One of ``"Client_Primary"``,
            ``"Client_Partner"``, or ``"Clients_Combined"``.
        data_income: Polars DataFrame containing income information, or ``None`` to
            clear the entry.

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_client_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_client_data_to_reactives(
        reactives_shiny,
        client_level,
        "Income",
        data_income,
    )


def save_clients_combined_data_to_reactives(
    reactives_shiny: Any,
    data_category: Any,
    data_value: Any,
) -> dict:
    """Save combined household data to the Clients_Combined sub-level in the reactives structure.

    Convenience wrapper around :func:`save_client_data_to_reactives` that fixes
    ``client_level`` to ``"Clients_Combined"``.  Only ``"Assets"``, ``"Goals"``,
    and ``"Income"`` are accepted as ``data_category`` values (``"Personal_Info"``
    is not valid at the combined level).

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_category: Data category key.  One of ``"Assets"``, ``"Goals"``,
            or ``"Income"``.
        data_value: Polars DataFrame containing the combined data, or ``None`` to
            clear the entry.

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If data_category is ``"Personal_Info"`` or any other validation
            fails (see :func:`save_client_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_client_data_to_reactives(
        reactives_shiny,
        "Clients_Combined",
        data_category,
        data_value,
    )


# =========================================================================
# JSON EXPORT — build client_info.json dict from Data_Clients reactives
# =========================================================================


def build_client_info_json_from_reactives(reactives_shiny: Any) -> dict[str, Any]:
    """Build the ``client_info.json`` dictionary from ``reactives_shiny``.

    Reads all client data from ``User_Inputs_Shiny`` (where the client subtabs
    store every individual field as a ``reactive.Value``) and assembles the
    nested dictionary expected by the Typst report template ``report_QWIM.typ``.

    The client subtabs (``subtab_personal_info``, ``subtab_assets``,
    ``subtab_goals``, ``subtab_income``) each write their individual values to
    ``User_Inputs_Shiny`` using keys of the form::

        Input_Tab_clients_Subtab_clients_<Section>_<client>_<Field>

    This function reads those keys directly instead of the ``Data_Clients``
    DataFrames, which are only populated by the optional
    :func:`save_client_data_to_reactives` helpers.

    The returned dictionary has the following structure::

        {
            "single_or_couple": "Single" | "Couple",
            "personal_info": {
                "primary": {
                    "name", "age_current", "age_retirement",
                    "age_income_starting", "status_marital", "gender",
                    "tolerance_risk", "state", "code_zip"
                },
                "partner": {same fields},
            },
            "assets": {
                "primary":  {"taxable", "tax_deferred", "tax_free", "total"},
                "partner":  {same},
                "combined": {summed from primary + partner},
            },
            "goals": {
                "primary":  {"essential", "important", "aspirational", "total"},
                "partner":  {same},
                "combined": {summed from primary + partner},
            },
            "income": {
                "primary":  {
                    "social_security", "pension", "annuity_existing",
                    "other", "total"
                },
                "partner":  {same},
                "combined": {summed from primary + partner},
            },
        }

    When a reactive value has not yet been populated, text fields default to
    ``""`` and numeric fields to ``0.0``.

    Args:
        reactives_shiny: Shared reactive state dictionary.

    Returns
    -------
        dict[str, Any]: Nested dictionary ready to be serialised to
        ``client_info.json``.
    """

    # -----------------------------------------------------------------------
    # Inner helpers: safely read one key from User_Inputs_Shiny
    # -----------------------------------------------------------------------
    def _get_ui(key: str) -> Any:
        """Return the unwrapped value for *key* from User_Inputs_Shiny, or None."""
        if not isinstance(reactives_shiny, dict):
            return None
        user_inputs = reactives_shiny.get("User_Inputs_Shiny")
        if not isinstance(user_inputs, dict):
            return None
        reactive_var = user_inputs.get(key)
        if reactive_var is None:
            return None
        if hasattr(reactive_var, "get"):
            try:
                return reactive_var.get()
            except Exception:
                return None
        return reactive_var

    def _str_ui(key: str, default: str = "") -> str:
        """Return string value for *key*, or *default* when absent."""
        val = _get_ui(key)
        return str(val) if val is not None else default

    def _flt_ui(key: str, default: float = 0.0) -> float:
        """Return float value for *key*, or *default* when absent."""
        val = _get_ui(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # -----------------------------------------------------------------------
    # Single_Or_Couple indicator (stored in Data_Clients, defaulting "Single")
    # -----------------------------------------------------------------------
    single_or_couple: str = "Single"
    try:
        single_or_couple = get_single_or_couple_from_reactives(reactives_shiny)
    except Exception as exc:
        _logger.debug("Could not retrieve Single_Or_Couple from reactives: %s", exc)

    # -----------------------------------------------------------------------
    # Personal-info builder (per investor prefix)
    # -----------------------------------------------------------------------
    def _int_ui(key: str, default: int = 0) -> int:
        """Return integer value for *key*, or *default* when absent.

        Handles values returned as float (e.g. 60.0 from Shiny numeric inputs)
        by truncating to int, which avoids writing ``"60.0"`` strings into JSON
        that later cause Typst ``int("60.0")`` to fail.
        """
        val = _get_ui(key)
        if val is None:
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    def _build_personal_info(prefix: str, default_name: str) -> dict[str, Any]:
        """Build personal-info dict for one investor.

        *prefix* is ``"client_Primary"`` or ``"client_Partner"``.
        Age fields use ``_int_ui`` so that Shiny float values (e.g. ``60.0``)
        are stored as JSON integers (``60``), avoiding downstream Typst errors.
        """
        base = f"Input_Tab_clients_Subtab_clients_Personal_Info_{prefix}"
        name = _str_ui(f"{base}_Name") or default_name
        return {
            "name": name,
            "age_current": _int_ui(f"{base}_Age_Current"),
            "age_retirement": _int_ui(f"{base}_Age_Retirement"),
            "age_income_starting": _int_ui(f"{base}_Age_Income_Starting"),
            "status_marital": _str_ui(f"{base}_Status_Marital"),
            "gender": _str_ui(f"{base}_Gender"),
            "tolerance_risk": _str_ui(f"{base}_Tolerance_Risk"),
            "state": _str_ui(f"{base}_State"),
            "code_zip": _str_ui(f"{base}_Code_Zip"),
        }

    # -----------------------------------------------------------------------
    # Assets builder (per investor prefix; combined = sum of both)
    # -----------------------------------------------------------------------
    def _build_assets(prefix: str) -> dict[str, float]:
        """Build assets dict for one investor prefix."""
        base = f"Input_Tab_clients_Subtab_clients_Assets_{prefix}_Assets"
        taxable = _flt_ui(f"{base}_Taxable")
        tax_deferred = _flt_ui(f"{base}_Tax_Deferred")
        tax_free = _flt_ui(f"{base}_Tax_Free")
        return {
            "taxable": taxable,
            "tax_deferred": tax_deferred,
            "tax_free": tax_free,
            "total": taxable + tax_deferred + tax_free,
        }

    def _build_assets_combined() -> dict[str, float]:
        p = _build_assets("client_Primary")
        q = _build_assets("client_Partner")
        return {
            "taxable": p["taxable"] + q["taxable"],
            "tax_deferred": p["tax_deferred"] + q["tax_deferred"],
            "tax_free": p["tax_free"] + q["tax_free"],
            "total": p["total"] + q["total"],
        }

    # -----------------------------------------------------------------------
    # Goals builder (per investor prefix; combined = sum of both)
    # -----------------------------------------------------------------------
    def _build_goals(prefix: str) -> dict[str, float]:
        """Build goals dict for one investor prefix."""
        base = f"Input_Tab_clients_Subtab_clients_Goals_{prefix}_Goal"
        essential = _flt_ui(f"{base}_Essential")
        important = _flt_ui(f"{base}_Important")
        aspirational = _flt_ui(f"{base}_Aspirational")
        return {
            "essential": essential,
            "important": important,
            "aspirational": aspirational,
            "total": essential + important + aspirational,
        }

    def _build_goals_combined() -> dict[str, float]:
        p = _build_goals("client_Primary")
        q = _build_goals("client_Partner")
        return {
            "essential": p["essential"] + q["essential"],
            "important": p["important"] + q["important"],
            "aspirational": p["aspirational"] + q["aspirational"],
            "total": p["total"] + q["total"],
        }

    # -----------------------------------------------------------------------
    # Income builder (per investor prefix; combined = sum of both)
    # -----------------------------------------------------------------------
    def _build_income(prefix: str) -> dict[str, float]:
        """Build income dict for one investor prefix."""
        base = f"Input_Tab_clients_Subtab_clients_Income_{prefix}_Income"
        ss = _flt_ui(f"{base}_Social_Security")
        pension = _flt_ui(f"{base}_Pension")
        annuity = _flt_ui(f"{base}_Annuity_Existing")
        other = _flt_ui(f"{base}_Other")
        return {
            "social_security": ss,
            "pension": pension,
            "annuity_existing": annuity,
            "other": other,
            "total": ss + pension + annuity + other,
        }

    def _build_income_combined() -> dict[str, float]:
        p = _build_income("client_Primary")
        q = _build_income("client_Partner")
        return {
            "social_security": p["social_security"] + q["social_security"],
            "pension": p["pension"] + q["pension"],
            "annuity_existing": p["annuity_existing"] + q["annuity_existing"],
            "other": p["other"] + q["other"],
            "total": p["total"] + q["total"],
        }

    # -----------------------------------------------------------------------
    # Assemble and return the complete dictionary
    # -----------------------------------------------------------------------
    _logger.info(
        "Building client_info JSON from User_Inputs_Shiny reactives",
        extra={"single_or_couple": single_or_couple},
    )

    return {
        "single_or_couple": single_or_couple,
        "personal_info": {
            "primary": _build_personal_info("client_Primary", "Primary Investor"),
            "partner": _build_personal_info("client_Partner", "Partner Investor"),
        },
        "assets": {
            "primary": _build_assets("client_Primary"),
            "partner": _build_assets("client_Partner"),
            "combined": _build_assets_combined(),
        },
        "goals": {
            "primary": _build_goals("client_Primary"),
            "partner": _build_goals("client_Partner"),
            "combined": _build_goals_combined(),
        },
        "income": {
            "primary": _build_income("client_Primary"),
            "partner": _build_income("client_Partner"),
            "combined": _build_income_combined(),
        },
    }


# =========================================================================
# DATA_RESULTS — constants, validation, core get/save, convenience wrappers
# =========================================================================

#: Valid subtab keys within the Data_Results structure
VALID_RESULTS_SUBTAB_KEYS: frozenset[str] = frozenset(
    [
        "Portfolio_Analysis_Inputs",
        "Portfolio_Analysis_Outputs",
        "Portfolio_Comparison_Inputs",
        "Portfolio_Comparison_Outputs",
        "Weights_Analysis_Inputs",
        "Weights_Analysis_Outputs",
        "Portfolio_Optimization_Skfolio_Inputs",
        "Portfolio_Optimization_Skfolio_Outputs",
        "Portfolio_Simulation_Inputs",
        "Portfolio_Simulation_Outputs",
    ],
)


def validate_data_results_in_reactives(reactives_shiny: Any) -> tuple[bool, str]:
    """Validate that the Data_Results category exists and has the required structure.

    Checks that ``reactives_shiny["Data_Results"]`` is present, is a dictionary,
    and contains all ten expected subtab keys.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.

    Returns
    -------
        tuple[bool, str]: ``(True, "")`` when the structure is valid, or
        ``(False, error_message)`` with a descriptive error message otherwise.
    """
    # Input validation with early returns
    if reactives_shiny is None:
        return False, "reactives_shiny dictionary cannot be None"

    if not isinstance(reactives_shiny, dict):
        return (
            False,
            f"reactives_shiny must be a dictionary, got {type(reactives_shiny).__name__}",
        )

    # Configuration validation - Data_Results category must exist
    if "Data_Results" not in reactives_shiny:
        available_categories = list(reactives_shiny.keys())
        return (
            False,
            f"'Data_Results' category not found in reactives_shiny. "
            f"Available categories: {available_categories}",
        )

    data_results = reactives_shiny["Data_Results"]

    if not isinstance(data_results, dict):
        return (
            False,
            f"'Data_Results' must be a dictionary, got {type(data_results).__name__}",
        )

    # Business logic validation - all subtab keys must be present
    for subtab_key_name in sorted(VALID_RESULTS_SUBTAB_KEYS):
        if subtab_key_name not in data_results:
            available_keys = list(data_results.keys())
            return (
                False,
                f"Subtab key '{subtab_key_name}' not found in Data_Results. "
                f"Available keys: {available_keys}",
            )

    return True, ""


def validate_results_subtab_key(subtab_key: Any) -> tuple[bool, str]:
    """Validate that the provided subtab key is one of the allowed Data_Results keys.

    Args:
        subtab_key: The subtab key name to validate.

    Returns
    -------
        tuple[bool, str]: ``(True, "")`` when valid, or
        ``(False, error_message)`` when invalid.
    """
    # Input validation with early returns
    if subtab_key is None:
        return False, "subtab_key cannot be None"

    if not isinstance(subtab_key, str):
        return False, f"subtab_key must be a string, got {type(subtab_key).__name__}"

    if len(subtab_key.strip()) == 0:
        return False, "subtab_key cannot be an empty string"

    # Business logic validation
    if subtab_key not in VALID_RESULTS_SUBTAB_KEYS:
        return (
            False,
            f"Invalid subtab_key '{subtab_key}'. Valid keys: {sorted(VALID_RESULTS_SUBTAB_KEYS)}",
        )

    return True, ""


def get_results_data_from_reactives(
    reactives_shiny: Any,
    subtab_key: Any,
) -> Any | None:
    """Retrieve results data stored at a specific subtab key in Data_Results.

    Accesses ``reactives_shiny["Data_Results"][subtab_key]`` and returns the
    current reactive value (a dict, Polars DataFrame, or ``None``).

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        subtab_key: One of the ten valid subtab keys in ``Data_Results`` (e.g.
            ``"Portfolio_Analysis_Inputs"``).

    Returns
    -------
        Any | None: The stored value, or ``None`` when no data has been saved yet
        for the given subtab key.

    Raises
    ------
        ValueError: If the reactives structure or subtab_key is invalid.
        KeyError: If the requested key is not found in Data_Results.
        RuntimeError: If the reactive value cannot be retrieved.
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_results_in_reactives(reactives_shiny)
    if not validation_result:
        error_message = f"Data_Results structure validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    validation_result, validation_message = validate_results_subtab_key(subtab_key)
    if not validation_result:
        error_message = f"Subtab key validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    # Configuration validation - subtab key must be present in Data_Results
    if subtab_key not in reactives_shiny["Data_Results"]:
        available_keys = list(reactives_shiny["Data_Results"].keys())
        error_message = (
            f"Subtab key '{subtab_key}' not found in Data_Results. Available keys: {available_keys}"
        )
        _logger.error(error_message)
        raise KeyError(error_message)

    reactive_variable = reactives_shiny["Data_Results"][subtab_key]

    # Business logic validation
    if reactive_variable is None:
        error_message = (
            f"Reactive variable at Data_Results['{subtab_key}'] is None — "
            f"it was not properly initialized"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    if not hasattr(reactive_variable, "get"):
        error_message = (
            f"Reactive variable at Data_Results['{subtab_key}'] "
            f"does not have a 'get' method. "
            f"Object type: {type(reactive_variable).__name__}"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    # Only use try-except for unpredictable reactive access
    try:
        retrieved_value = reactive_variable.get()
        _logger.debug(
            "Retrieved results data from reactives",
            extra={
                "subtab_key": subtab_key,
                "value_type": type(retrieved_value).__name__,
            },
        )
        return retrieved_value

    except Exception as exc:
        error_message = f"Unexpected error retrieving Data_Results['{subtab_key}']: {exc}"
        _logger.error(error_message)
        raise RuntimeError(error_message) from exc


def save_results_data_to_reactives(
    reactives_shiny: Any,
    subtab_key: Any,
    data_value: Any,
) -> dict:
    """Save results data to a specific subtab key in the Data_Results reactive structure.

    Sets ``reactives_shiny["Data_Results"][subtab_key]`` to the provided value.
    Any Python object (dict, Polars DataFrame, list, ``None``) is accepted so
    that each subtab is free to store whichever shape of data is most natural.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        subtab_key: One of the ten valid subtab keys in ``Data_Results`` (e.g.
            ``"Portfolio_Analysis_Inputs"``).
        data_value: Value to store.  Pass ``None`` to clear a previously stored value.

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If the reactives structure or subtab_key is invalid.
        KeyError: If the requested key is not found in Data_Results.
        RuntimeError: If the reactive value cannot be updated.
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_results_in_reactives(reactives_shiny)
    if not validation_result:
        error_message = f"Data_Results structure validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    validation_result, validation_message = validate_results_subtab_key(subtab_key)
    if not validation_result:
        error_message = f"Subtab key validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    # Configuration validation - subtab key must be present in Data_Results
    if subtab_key not in reactives_shiny["Data_Results"]:
        available_keys = list(reactives_shiny["Data_Results"].keys())
        error_message = (
            f"Subtab key '{subtab_key}' not found in Data_Results. Available keys: {available_keys}"
        )
        _logger.error(error_message)
        raise KeyError(error_message)

    reactive_variable = reactives_shiny["Data_Results"][subtab_key]

    if reactive_variable is None:
        error_message = (
            f"Reactive variable at Data_Results['{subtab_key}'] is None — "
            f"it was not properly initialized"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    if not hasattr(reactive_variable, "set"):
        error_message = (
            f"Reactive variable at Data_Results['{subtab_key}'] "
            f"does not have a 'set' method. "
            f"Object type: {type(reactive_variable).__name__}"
        )
        _logger.error(error_message)
        raise ValueError(error_message)

    # Only use try-except for unpredictable reactive access
    try:
        reactive_variable.set(data_value)
        _logger.debug(
            "Saved results data to reactives",
            extra={
                "subtab_key": subtab_key,
                "value_type": type(data_value).__name__,
            },
        )
        return reactives_shiny

    except Exception as exc:
        error_message = f"Unexpected error saving Data_Results['{subtab_key}']: {exc}"
        _logger.error(error_message)
        raise RuntimeError(error_message) from exc


# -------------------------------------------------------------------------
# Convenience wrappers — one save function per subtab key
# -------------------------------------------------------------------------


def save_portfolio_analysis_inputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save Portfolio Analysis subtab input values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Portfolio_Analysis_Inputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Input data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Portfolio_Analysis_Inputs",
        data_value,
    )


def save_portfolio_analysis_outputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save Portfolio Analysis subtab output values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Portfolio_Analysis_Outputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Output data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Portfolio_Analysis_Outputs",
        data_value,
    )


def save_portfolio_comparison_inputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save Portfolio Comparison subtab input values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Portfolio_Comparison_Inputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Input data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Portfolio_Comparison_Inputs",
        data_value,
    )


def save_portfolio_comparison_outputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save Portfolio Comparison subtab output values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Portfolio_Comparison_Outputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Output data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Portfolio_Comparison_Outputs",
        data_value,
    )


def save_weights_analysis_inputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save Weights Analysis subtab input values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Weights_Analysis_Inputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Input data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Weights_Analysis_Inputs",
        data_value,
    )


def save_weights_analysis_outputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save Weights Analysis subtab output values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Weights_Analysis_Outputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Output data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Weights_Analysis_Outputs",
        data_value,
    )


def save_portfolio_optimization_skfolio_inputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save skfolio Optimization subtab input values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Portfolio_Optimization_Skfolio_Inputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Input data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Portfolio_Optimization_Skfolio_Inputs",
        data_value,
    )


def save_portfolio_optimization_skfolio_outputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save skfolio Optimization subtab output values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Portfolio_Optimization_Skfolio_Outputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Output data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Portfolio_Optimization_Skfolio_Outputs",
        data_value,
    )


def save_portfolio_simulation_inputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save Portfolio Simulation subtab input values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Portfolio_Simulation_Inputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Input data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Portfolio_Simulation_Inputs",
        data_value,
    )


def save_portfolio_simulation_outputs_to_reactives(
    reactives_shiny: Any,
    data_value: Any,
) -> dict:
    """Save Portfolio Simulation subtab output values to the Data_Results reactive structure.

    Convenience wrapper around :func:`save_results_data_to_reactives` that fixes
    ``subtab_key`` to ``"Portfolio_Simulation_Outputs"``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        data_value: Output data to store (dict, Polars DataFrame, or ``None``).

    Returns
    -------
        dict: The updated reactives_shiny dictionary.

    Raises
    ------
        ValueError: If any validation fails (see :func:`save_results_data_to_reactives`).
        RuntimeError: If the reactive value cannot be updated.
    """
    return save_results_data_to_reactives(
        reactives_shiny,
        "Portfolio_Simulation_Outputs",
        data_value,
    )


# =========================================================================
# JSON EXPORT — build Data_Results dicts from reactives for report export
# =========================================================================


def _coerce_results_value_to_json(data: Any) -> dict[str, Any]:
    """Convert a raw ``Data_Results`` value to a JSON-serialisable dict.

    Handles the common types that subtab server functions store in
    ``Data_Results``:

    * ``None`` → empty dict ``{}``
    * :class:`polars.DataFrame` → ``{"records": [<row-dicts>]}``
    * ``dict`` → returned as-is (values must already be serialisable)
    * ``list`` → ``{"records": <list>}``
    * any other type → ``{"value": str(data)}``

    Args:
        data: Raw value retrieved from a ``reactive.Value`` inside ``Data_Results``.

    Returns
    -------
        dict[str, Any]: JSON-serialisable dictionary.
    """
    # Input validation with early returns
    if data is None:
        return {}

    if isinstance(data, pl.DataFrame):
        return {"records": data.to_dicts()}

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        return {"records": data}

    # Fallback — convert unknown types to a string representation
    return {"value": str(data)}


def build_results_data_json_from_reactives(
    reactives_shiny: Any,
    subtab_key: Any,
) -> dict[str, Any]:
    """Build a JSON-serialisable dict from ``Data_Results`` for one subtab key.

    Retrieves the value stored at
    ``reactives_shiny["Data_Results"][subtab_key]`` and converts it to a
    JSON-serialisable dictionary.  The conversion follows these rules:

    * ``None`` → ``{}``
    * :class:`polars.DataFrame` → ``{"records": [<row-dicts>]}``
    * ``dict`` → returned as-is
    * ``list`` → ``{"records": <list>}``
    * any other type → ``{"value": str(data)}``

    This mirrors the :func:`build_client_info_json_from_reactives` pattern used
    for ``Data_Clients`` and is the canonical function called by
    ``report_data_export`` export functions writing to ``inputs_json/`` and
    ``outputs_json/``.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category.
        subtab_key: Key identifying the subtab result within ``Data_Results``.
            Must be one of the strings in :data:`VALID_RESULTS_SUBTAB_KEYS`.

    Returns
    -------
        dict[str, Any]: JSON-serialisable dictionary representing the stored
        subtab data, or an empty dict when no data has been saved yet.

    Raises
    ------
        ValueError: If the ``reactives_shiny`` structure is invalid or
            ``subtab_key`` is not a recognised key.
        RuntimeError: If the reactive value cannot be retrieved.
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_results_in_reactives(reactives_shiny)
    if not validation_result:
        _logger.error(
            "build_results_data_json_from_reactives: invalid reactives structure: %s",
            validation_message,
        )
        raise ValueError(
            f"build_results_data_json_from_reactives: invalid reactives structure: "
            f"{validation_message}",
        )

    validation_result, validation_message = validate_results_subtab_key(subtab_key)
    if not validation_result:
        _logger.error(
            "build_results_data_json_from_reactives: invalid subtab_key: %s",
            validation_message,
        )
        raise ValueError(
            f"build_results_data_json_from_reactives: invalid subtab_key: {validation_message}",
        )

    # Retrieve raw reactive value (may be None when subtab not yet populated)
    # Only use try-except for unpredictable reactive access
    try:
        raw_data = get_results_data_from_reactives(reactives_shiny, subtab_key)
    except RuntimeError as exc:
        _logger.warning(
            "build_results_data_json_from_reactives: could not retrieve '%s' "
            "from reactives, returning empty dict: %s",
            subtab_key,
            exc,
        )
        return {}

    # Business logic: coerce raw value to a JSON-serialisable dict
    json_dict = _coerce_results_value_to_json(raw_data)

    _logger.info(
        "build_results_data_json_from_reactives: built JSON dict for '%s' (%d top-level keys)",
        subtab_key,
        len(json_dict),
    )

    return json_dict


# =============================================================================
# VISUAL OBJECTS — SVG export from Visual_Objects_Shiny reactive state
# =============================================================================

#: Mapping from Visual_Objects_Shiny reactive key to output SVG filename.
_VISUAL_OBJECT_SVG_FILENAMES: dict[str, str] = {
    "Chart_Weights_Analysis": "weights_analysis.svg",
    "Charts_Weights_Composition": "weights_composition.svg",
    "Chart_Portfolio_Analysis": "portfolio_analysis.svg",
    "Chart_Portfolio_Comparison": "portfolio_comparison.svg",
    "Chart_Skfolio_Weights": "skfolio_weights.svg",
    "Chart_Skfolio_Performance": "skfolio_performance.svg",
    "Chart_Simulation_Fan": "simulation_fan_chart.svg",
    "Chart_Simulation_Histogram": "simulation_histogram.svg",
}

#: Default output directory for SVG files.
_OUTPUTS_IMAGES_DIR: Path = Path(__file__).resolve().parents[1] / "reporting" / "outputs_images"


def export_visual_objects_to_svg(
    reactives_shiny: Any,
    output_dir: Path | None = None,
) -> dict[str, Path | None]:
    """Save Plotly figures stored in ``Visual_Objects_Shiny`` as SVG image files.

    Called by the report pipeline immediately before PDF compilation so that
    the Typst template can embed the latest live charts instead of the static
    plotnine fallbacks.

    Parameters
    ----------
    reactives_shiny : Any
        Shared reactive state dictionary produced by
        ``initialize_reactives_shiny()``.
    output_dir : Path | None
        Directory in which to write the SVG files.  Defaults to
        ``src/dashboard/reporting/outputs_images/``.

    Returns
    -------
    dict[str, Path | None]
        Mapping of SVG *filename* to the saved :class:`~pathlib.Path`, or
        ``None`` when a chart could not be exported (figure absent or kaleido
        unavailable).
    """
    if output_dir is None:
        output_dir = _OUTPUTS_IMAGES_DIR

    # Input validation with early returns
    if reactives_shiny is None or not isinstance(reactives_shiny, dict):
        _logger.warning("export_visual_objects_to_svg: invalid reactives_shiny — skipping export")
        return dict.fromkeys(_VISUAL_OBJECT_SVG_FILENAMES.values())

    visual_objects = reactives_shiny.get("Visual_Objects_Shiny")
    if not isinstance(visual_objects, dict):
        _logger.warning(
            "export_visual_objects_to_svg: 'Visual_Objects_Shiny' missing or not a dict — skipping",
        )
        return dict.fromkeys(_VISUAL_OBJECT_SVG_FILENAMES.values())

    # Ensure output directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _logger.error(
            "export_visual_objects_to_svg: cannot create output dir '%s': %s",
            output_dir,
            exc,
        )
        return dict.fromkeys(_VISUAL_OBJECT_SVG_FILENAMES.values())

    results: dict[str, Path | None] = {}

    for chart_key, svg_filename in _VISUAL_OBJECT_SVG_FILENAMES.items():
        svg_path = output_dir / svg_filename
        results[svg_filename] = None

        reactive_value = visual_objects.get(chart_key)
        if reactive_value is None:
            _logger.debug(
                "export_visual_objects_to_svg: key '%s' not found in Visual_Objects_Shiny",
                chart_key,
            )
            continue

        # Retrieve the stored Plotly figure
        try:
            fig = (
                reactive_value.get() if hasattr(reactive_value, "get") else reactive_value
            )  # plain value (defensive)
        except Exception as exc:
            _logger.warning(
                "export_visual_objects_to_svg: could not read reactive value for '%s': %s",
                chart_key,
                exc,
            )
            continue

        if fig is None:
            _logger.debug(
                "export_visual_objects_to_svg: no figure stored yet for '%s'",
                chart_key,
            )
            continue

        # Write SVG using Plotly / kaleido
        try:
            fig.write_image(str(svg_path), format="svg", width=1200, height=700)
            results[svg_filename] = svg_path
            _logger.info(
                "export_visual_objects_to_svg: saved '%s' (%d bytes)",
                svg_filename,
                svg_path.stat().st_size,
            )
        except Exception as exc:
            # kaleido not installed, or figure is not a Plotly go.Figure
            _logger.warning(
                "export_visual_objects_to_svg: could not export '%s' to SVG: %s",
                chart_key,
                exc,
            )

    saved_count = sum(1 for v in results.values() if v is not None)
    _logger.info(
        "export_visual_objects_to_svg: exported %d / %d charts to '%s'",
        saved_count,
        len(results),
        output_dir,
    )
    return results
