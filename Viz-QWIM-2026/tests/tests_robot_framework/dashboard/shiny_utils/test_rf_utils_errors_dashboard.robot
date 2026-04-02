*** Settings ***
Documentation
...    Robot Framework tests for utils_errors_dashboard helpers.
...
...    Tests cover:
...    - is_silent_exception behaviour for various exception types
...    - DeprecationWarning emission from deprecated exception aliases
...
...    All tests run against pure-Python logic — no Shiny server required.
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-05-01

Library     ${CURDIR}${/}keywords_utils_tab_portfolios_clients.py
Library     Collections

Suite Setup      Log    Starting Utils Errors Dashboard RF Test Suite
Suite Teardown   Log    Utils Errors Dashboard RF Test Suite Complete


*** Test Cases ***

# ---------------------------------------------------------------------------
# is_silent_exception
# ---------------------------------------------------------------------------

AttributeError Is A Silent Exception
    [Documentation]    is_silent_exception(AttributeError) returns True.
    [Tags]    utils_errors_dashboard    is_silent_exception    smoke
    ${result}=    Check Attribute Error Is Silent
    Should Be Equal    ${result}    ${True}

ValueError Is Not A Silent Exception
    [Documentation]    is_silent_exception(ValueError) returns False.
    [Tags]    utils_errors_dashboard    is_silent_exception    smoke
    ${result}=    Check Value Error Is Not Silent
    Should Be Equal    ${result}    ${False}

# ---------------------------------------------------------------------------
# DeprecationWarning from deprecated aliases
# ---------------------------------------------------------------------------

Error Silent Initialization Emits Deprecation Warning
    [Documentation]    Instantiating Error_Silent_Initialization emits DeprecationWarning.
    [Tags]    utils_errors_dashboard    deprecation_warning    smoke
    ${result}=    Instantiate Error Silent Initialization Emits Deprecation
    Should Be Equal    ${result}    ${True}

Error Dashboard Initialization Emits Deprecation Warning
    [Documentation]    Instantiating Error_Dashboard_Initialization emits DeprecationWarning.
    [Tags]    utils_errors_dashboard    deprecation_warning
    ${result}=    Instantiate Error Dashboard Initialization Emits Deprecation
    Should Be Equal    ${result}    ${True}
