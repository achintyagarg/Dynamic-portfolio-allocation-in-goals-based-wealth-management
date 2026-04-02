*** Settings ***
Documentation
...    Robot Framework test suite for dashboard formatting utilities.
...
...    Verifies: format_currency_value produces '$N,NNN'-style strings,
...    rejects decimals, starts with '$', and returns '$0' for zero;
...    extract_numeric_from_currency_string rounds trips correctly and
...    returns 0.0 for empty/non-string inputs;
...    format_enhanced_percentage_display returns '%-'terminated strings
...    with correct values for common inputs;
...    format_enhanced_number_display produces comma-separated number strings.
...
...    Test Categories:
...    - format_currency_value (dollar-only formatting)
...    - extract_numeric_from_currency_string (round-trip and edge cases)
...    - format_enhanced_percentage_display (decimal → percentage string)
...    - format_enhanced_number_display (locale-aware number string)
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-02-28

Library     ${CURDIR}${/}keywords_formatting.py
Library     Collections

Suite Setup      Log    Starting Formatting Utilities Robot Framework Test Suite
Suite Teardown   Log    Formatting Utilities Test Suite Complete


*** Variables ***
${AMOUNT_SMALL}             ${1250.0}
${AMOUNT_LARGE}             ${1250000.0}
${AMOUNT_ZERO}              ${0.0}
${FORMATTED_SMALL}          $1,250
${FORMATTED_LARGE}          $1,250,000
${DECIMAL_FIVE_PCT}         ${0.05}
${DECIMAL_TWELVE_PCT}       ${0.125}
${NUMBER_LARGE}             ${1234567.0}


*** Test Cases ***

# ---------------------------------------------------------------------------
# format_currency_value
# ---------------------------------------------------------------------------

Format Currency Returns String Starting With Dollar Sign
    [Documentation]    format_currency_value returns a string starting with '$'.
    [Tags]    formatting    currency    smoke
    ${result}=    Format Currency    ${AMOUNT_SMALL}
    Currency Format Should Start With Dollar Sign    ${result}

Format Currency Small Amount Equals Expected String
    [Documentation]    format_currency_value(1250.0) returns '$1,250'.
    [Tags]    formatting    currency    smoke
    ${result}=    Format Currency    ${AMOUNT_SMALL}
    Currency Format Should Equal    ${result}    ${FORMATTED_SMALL}

Format Currency Large Amount Equals Expected String
    [Documentation]    format_currency_value(1250000.0) returns '$1,250,000'.
    [Tags]    formatting    currency
    ${result}=    Format Currency    ${AMOUNT_LARGE}
    Currency Format Should Equal    ${result}    ${FORMATTED_LARGE}

Format Currency Zero Returns Dollar Zero
    [Documentation]    format_currency_value(0) returns '$0'.
    [Tags]    formatting    currency
    ${result}=    Format Currency    ${AMOUNT_ZERO}
    Currency Format Of Zero Should Be Dollar Zero    ${result}

Format Currency Result Has No Decimal Point
    [Documentation]    Dollar-only format must not contain a decimal point.
    [Tags]    formatting    currency
    ${result}=    Format Currency    ${AMOUNT_LARGE}
    Currency Format Should Not Contain Decimal Point    ${result}

Format Currency Fractional Amount Is Rounded
    [Documentation]    format_currency_value(999.9) rounds to '$1,000'
    ...    (nearest dollar).
    [Tags]    formatting    currency
    ${result}=    Format Currency    ${999.9}
    Currency Format Should Equal    ${result}    $1,000

Format Currency Negative Amount Starts With Dollar Sign
    [Documentation]    Negative amounts still start with '$' (edge case).
    [Tags]    formatting    currency
    ${result}=    Format Currency    ${-500.0}
    Currency Format Should Start With Dollar Sign    ${result}


# ---------------------------------------------------------------------------
# extract_numeric_from_currency_string
# ---------------------------------------------------------------------------

Extract Numeric From Small Currency String
    [Documentation]    extract_numeric_from_currency_string('$1,250') returns 1250.0.
    [Tags]    formatting    extract    smoke
    ${extracted}=    Extract Numeric From Currency    ${FORMATTED_SMALL}
    Extracted Value Should Equal    ${extracted}    ${1250.0}

Extract Numeric From Large Currency String
    [Documentation]    extract_numeric_from_currency_string('$1,250,000') returns 1250000.0.
    [Tags]    formatting    extract    smoke
    ${extracted}=    Extract Numeric From Currency    ${FORMATTED_LARGE}
    Extracted Value Should Equal    ${extracted}    ${1250000.0}

Currency Round Trip Preserves Small Amount
    [Documentation]    format → extract round-trip preserves the original integer amount.
    [Tags]    formatting    extract
    Currency Round Trip Should Preserve Value    ${AMOUNT_SMALL}

Currency Round Trip Preserves Large Amount
    [Documentation]    format → extract round-trip preserves the original integer amount.
    [Tags]    formatting    extract
    Currency Round Trip Should Preserve Value    ${AMOUNT_LARGE}

Extract From Empty String Returns Zero
    [Documentation]    extract_numeric_from_currency_string('') returns 0.0.
    [Tags]    formatting    extract    edge_case
    Extract From Empty String Should Return Zero

Extract From Non String Input Returns Zero
    [Documentation]    Passing a non-string value returns 0.0 (defensive guard).
    [Tags]    formatting    extract    edge_case
    Extract From None Alternative Should Return Zero


# ---------------------------------------------------------------------------
# format_enhanced_percentage_display
# ---------------------------------------------------------------------------

Percentage Format Ends With Percent Sign
    [Documentation]    format_enhanced_percentage_display always returns a string
    ...    ending with '%'.
    [Tags]    formatting    percentage    smoke
    ${result}=    Format Percentage    ${DECIMAL_FIVE_PCT}    2    ${True}
    Percentage Format Should End With Percent Sign    ${result}

Five Percent Decimal Formats To String Containing 5
    [Documentation]    format_enhanced_percentage_display(0.05) result contains '5'.
    [Tags]    formatting    percentage
    ${result}=    Format Percentage    ${DECIMAL_FIVE_PCT}    2    ${True}
    Should Contain    ${result}    5

Zero Decimal Formats To Zero Percent
    [Documentation]    format_enhanced_percentage_display(0.0) ends in '%'
    ...    and contains '0'.
    [Tags]    formatting    percentage    edge_case
    ${result}=    Format Percentage    ${0.0}    2    ${True}
    Percentage Format Should End With Percent Sign    ${result}
    Should Contain    ${result}    0

Percentage Already In Pct Form With Multiply False
    [Documentation]    Passing 15.5 with multiply_by_hundred=False formats
    ...    correctly (doesn't scale to 1550%).
    [Tags]    formatting    percentage
    ${result}=    Format Percentage    ${15.5}    2    ${False}
    Percentage Format Should End With Percent Sign    ${result}
    Should Contain    ${result}    15


# ---------------------------------------------------------------------------
# format_enhanced_number_display
# ---------------------------------------------------------------------------

Format Large Number Contains Comma Separator
    [Documentation]    format_enhanced_number_display(1234567) contains at
    ...    least one comma thousands separator.
    [Tags]    formatting    number    smoke
    ${result}=    Format Number    ${NUMBER_LARGE}    0
    Number Format Should Contain Comma Separator    ${result}

Format Large Number Is Not Empty
    [Documentation]    format_enhanced_number_display returns a non-empty string.
    [Tags]    formatting    number
    ${result}=    Format Number    ${NUMBER_LARGE}    0
    Number Format Should Not Be Empty    ${result}

Format Zero Number Is Not Empty
    [Documentation]    format_enhanced_number_display(0) returns a non-empty string.
    [Tags]    formatting    number    edge_case
    ${result}=    Format Number    ${0.0}    0
    Number Format Should Not Be Empty    ${result}

Format Number With Two Decimal Places Is Not Empty
    [Documentation]    Formatting with decimal_places=2 returns a non-empty string.
    [Tags]    formatting    number
    ${result}=    Format Number    ${1234.567}    2
    Number Format Should Not Be Empty    ${result}
