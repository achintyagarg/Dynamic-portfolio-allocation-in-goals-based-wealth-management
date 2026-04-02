*** Settings ***
Documentation
...    Robot Framework tests for utils_tab_portfolios.validate_portfolio_data
...    and utils_tab_clients utility helpers.
...
...    All tests run against pure-Python logic — no Shiny server required.
...
...    Test categories:
...    - Portfolio data validation (validate_portfolio_data)
...    - Currency formatting (format_currency_display)
...    - Financial amount validation (validate_financial_amount)
...    - Age range validation (validate_age_range)
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-05-01

Library     ${CURDIR}${/}keywords_utils_tab_portfolios_clients.py
Library     Collections

Suite Setup      Log    Starting Utils Tab Portfolios + Clients RF Test Suite
Suite Teardown   Log    Utils Tab Portfolios + Clients Test Suite Complete


*** Test Cases ***

# ---------------------------------------------------------------------------
# validate_portfolio_data — rejection cases
# ---------------------------------------------------------------------------

None Portfolio Data Is Rejected
    [Documentation]    validate_portfolio_data(None) returns (False, non-empty message).
    [Tags]    utils_tab_portfolios    validation    smoke
    ${result}=    Validate Portfolio Data With None
    ${is_valid}=    Result Is Valid    ${result}
    Should Be Equal    ${is_valid}    ${False}
    ${msg}=    Result Message    ${result}
    Should Not Be Empty    ${msg}

Empty DataFrame Is Rejected
    [Documentation]    validate_portfolio_data(empty df) returns (False, ...).
    [Tags]    utils_tab_portfolios    validation
    ${result}=    Validate Portfolio Data With Empty Df
    ${is_valid}=    Result Is Valid    ${result}
    Should Be Equal    ${is_valid}    ${False}

DataFrame Without Date Column Is Rejected
    [Documentation]    DataFrame missing Date column → (False, message with 'Date').
    [Tags]    utils_tab_portfolios    validation
    ${result}=    Validate Portfolio Data Without Date Column
    ${is_valid}=    Result Is Valid    ${result}
    Should Be Equal    ${is_valid}    ${False}
    ${msg}=    Result Message    ${result}
    Should Contain    ${msg}    Date

DataFrame Without Value Column Is Rejected For Portfolio Data
    [Documentation]    portfolio data without Value column → (False, message with 'Value').
    [Tags]    utils_tab_portfolios    validation
    ${result}=    Validate Portfolio Data Without Value Column
    ${is_valid}=    Result Is Valid    ${result}
    Should Be Equal    ${is_valid}    ${False}
    ${msg}=    Result Message    ${result}
    Should Contain    ${msg}    Value

All Non-Numeric Value Column Is Rejected
    [Documentation]    All-string Value column → (False, message mentioning 'numeric').
    [Tags]    utils_tab_portfolios    validation
    ${result}=    Validate Portfolio Data With All Non Numeric Values
    ${is_valid}=    Result Is Valid    ${result}
    Should Be Equal    ${is_valid}    ${False}
    ${msg}=    Result Message    ${result}
    Should Contain    ${msg}    numeric

Weights Data Without Components Is Rejected
    [Documentation]    weights data with only Date column → (False, ...).
    [Tags]    utils_tab_portfolios    validation
    ${result}=    Validate Weights Data Without Components
    ${is_valid}=    Result Is Valid    ${result}
    Should Be Equal    ${is_valid}    ${False}

# ---------------------------------------------------------------------------
# validate_portfolio_data — acceptance cases
# ---------------------------------------------------------------------------

Valid Portfolio Data Is Accepted
    [Documentation]    Fully valid portfolio DataFrame → (True, '').
    [Tags]    utils_tab_portfolios    validation    smoke
    ${result}=    Validate Portfolio Data With Valid Data
    ${is_valid}=    Result Is Valid    ${result}
    Should Be Equal    ${is_valid}    ${True}
    ${msg}=    Result Message    ${result}
    Should Be Empty    ${msg}

Valid Weights Data With Components Is Accepted
    [Documentation]    Weights DataFrame with component columns → (True, '').
    [Tags]    utils_tab_portfolios    validation
    ${result}=    Validate Weights Data With Components
    ${is_valid}=    Result Is Valid    ${result}
    Should Be Equal    ${is_valid}    ${True}

# ---------------------------------------------------------------------------
# format_currency_display
# ---------------------------------------------------------------------------

Format Zero Returns Dollar Zero
    [Documentation]    format_currency_display(0) == "$0"
    [Tags]    utils_tab_clients    formatting    smoke
    ${result}=    Format Zero As Currency
    Should Be Equal    ${result}    $0

Format One Million Returns Formatted String
    [Documentation]    format_currency_display(1_000_000) == "$1,000,000"
    [Tags]    utils_tab_clients    formatting
    ${result}=    Format One Million As Currency
    Should Be Equal    ${result}    $1,000,000

Format None Returns Dollar Zero
    [Documentation]    format_currency_display(None) == "$0"
    [Tags]    utils_tab_clients    formatting
    ${result}=    Format None As Currency
    Should Be Equal    ${result}    $0

# ---------------------------------------------------------------------------
# validate_financial_amount
# ---------------------------------------------------------------------------

Zero Financial Amount Is Valid
    [Documentation]    validate_financial_amount(0) == 0.0
    [Tags]    utils_tab_clients    validation
    ${result}=    Validate Zero Financial Amount
    Should Be Equal As Numbers    ${result}    0.0

Positive Financial Amount Is Valid
    [Documentation]    validate_financial_amount(250000) == 250000.0
    [Tags]    utils_tab_clients    validation
    ${result}=    Validate Positive Financial Amount    250000.0
    Should Be Equal As Numbers    ${result}    250000.0

None Financial Amount Becomes Zero
    [Documentation]    validate_financial_amount(None) == 0.0
    [Tags]    utils_tab_clients    validation
    ${result}=    Validate None Financial Amount
    Should Be Equal As Numbers    ${result}    0.0

Negative Financial Amount Raises ValueError
    [Documentation]    validate_financial_amount(-100) raises ValueError with 'negative'.
    [Tags]    utils_tab_clients    validation
    ${exc_msg}=    Validate Negative Financial Amount Raises
    Should Not Be Empty    ${exc_msg}
    Should Contain    ${exc_msg}    negative

# ---------------------------------------------------------------------------
# validate_age_range
# ---------------------------------------------------------------------------

Valid Age Within Range Is Accepted
    [Documentation]    validate_age_range(45, 18, 100) == 45
    [Tags]    utils_tab_clients    validation
    ${result}=    Validate Valid Age    45
    Should Be Equal As Integers    ${result}    45

Age Below Minimum Raises ValueError
    [Documentation]    validate_age_range(17, 18, 100) → ValueError with 'between'.
    [Tags]    utils_tab_clients    validation
    ${exc_msg}=    Validate Below Minimum Age Raises
    Should Not Be Empty    ${exc_msg}
    Should Contain    ${exc_msg}    between
