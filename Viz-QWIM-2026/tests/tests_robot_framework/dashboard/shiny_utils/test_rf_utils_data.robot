*** Settings ***
Documentation
...    Robot Framework test suite for dashboard utils_data (shiny_utils).
...
...    Verifies: get_input_data_raw loads the three expected DataFrames from
...    inputs/raw/; validate_portfolio_data accepts valid 2-column DataFrames
...    and rejects None, wrong columns, and extra columns;
...    calculate_portfolio_returns returns a non-empty Series.
...
...    Test Categories:
...    - Raw input data loading (get_input_data_raw)
...    - Portfolio data validation (validate_portfolio_data)
...    - Portfolio returns calculation (calculate_portfolio_returns)
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-02-28

Library     ${CURDIR}${/}keywords_utils_data.py
Library     Collections

Suite Setup      Log    Starting Utils Data Robot Framework Test Suite
Suite Teardown   Log    Utils Data Test Suite Complete


*** Variables ***
${KEY_TIMESERIES}       Time_Series_Sample
${KEY_ETF}              Time_Series_ETFs
${KEY_WEIGHTS}          Weights_My_Portfolio


*** Test Cases ***

# ---------------------------------------------------------------------------
# get_input_data_raw
# ---------------------------------------------------------------------------

Raw Input Data Loads Without Error
    [Documentation]    get_input_data_raw returns a dict without raising an
    ...    exception (CSV files present in inputs/raw/).
    [Tags]    utils_data    raw_data    smoke
    ${data}=    Load Raw Input Data
    Should Not Be Equal    ${data}    ${None}

Raw Input Data Contains Time Series Sample Key
    [Documentation]    The returned dict contains the 'Time_Series_Sample' key.
    [Tags]    utils_data    raw_data
    ${data}=    Load Raw Input Data
    Raw Input Data Should Contain Key    ${data}    ${KEY_TIMESERIES}

Raw Input Data Contains Time Series ETFs Key
    [Documentation]    The returned dict contains the 'Time_Series_ETFs' key.
    [Tags]    utils_data    raw_data
    ${data}=    Load Raw Input Data
    Raw Input Data Should Contain Key    ${data}    ${KEY_ETF}

Raw Input Data Contains Weights Portfolio Key
    [Documentation]    The returned dict contains the 'Weights_My_Portfolio' key.
    [Tags]    utils_data    raw_data
    ${data}=    Load Raw Input Data
    Raw Input Data Should Contain Key    ${data}    ${KEY_WEIGHTS}

Time Series Sample Is Non-Empty DataFrame
    [Documentation]    data['Time_Series_Sample'] is a non-empty pl.DataFrame.
    [Tags]    utils_data    raw_data
    ${data}=    Load Raw Input Data
    Raw Input Data Value Should Be Non Empty Dataframe    ${data}    ${KEY_TIMESERIES}

Time Series ETFs Is Non-Empty DataFrame
    [Documentation]    data['Time_Series_ETFs'] is a non-empty pl.DataFrame.
    [Tags]    utils_data    raw_data
    ${data}=    Load Raw Input Data
    Raw Input Data Value Should Be Non Empty Dataframe    ${data}    ${KEY_ETF}

Weights Portfolio Is Non-Empty DataFrame
    [Documentation]    data['Weights_My_Portfolio'] is a non-empty pl.DataFrame.
    [Tags]    utils_data    raw_data
    ${data}=    Load Raw Input Data
    Raw Input Data Value Should Be Non Empty Dataframe    ${data}    ${KEY_WEIGHTS}


# ---------------------------------------------------------------------------
# validate_portfolio_data — valid input
# ---------------------------------------------------------------------------

Valid Two Column Portfolio DataFrame Passes Validation
    [Documentation]    validate_portfolio_data with a correctly shaped 2-column
    ...    (Date, Value) DataFrame returns (True, '').
    [Tags]    utils_data    validation    smoke
    ${result}=    Validate Valid Portfolio Dataframe
    Validation Result Should Be Valid    ${result}

Valid Portfolio Validation Returns True As First Element
    [Documentation]    The first element of the returned tuple is True for valid input.
    [Tags]    utils_data    validation
    ${result}=    Validate Valid Portfolio Dataframe
    ${is_valid}=    Set Variable    ${result[0]}
    Should Be True    ${is_valid}


# ---------------------------------------------------------------------------
# validate_portfolio_data — invalid inputs
# ---------------------------------------------------------------------------

None Portfolio Data Returns Invalid Validation Result
    [Documentation]    validate_portfolio_data(None) returns (False, error_msg).
    [Tags]    utils_data    validation
    ${result}=    Validate None Portfolio Data
    Validation Result Should Be Invalid    ${result}

None Portfolio Data Has Non-Empty Error Message
    [Documentation]    Error message is non-empty when portfolio data is None.
    [Tags]    utils_data    validation
    ${result}=    Validate None Portfolio Data
    Validation Error Message Should Not Be Empty    ${result}

Wrong Column Name Returns Invalid Validation Result
    [Documentation]    DataFrame with Date + wrong column name fails validation.
    [Tags]    utils_data    validation
    ${result}=    Validate Portfolio Data With Wrong Columns
    Validation Result Should Be Invalid    ${result}

Extra Column Returns Invalid Validation Result
    [Documentation]    3-column DataFrame (Date, Value, Extra) fails validation
    ...    because validate_portfolio_data requires exactly 2 columns.
    [Tags]    utils_data    validation
    ${result}=    Validate Portfolio Data With Extra Column
    Validation Result Should Be Invalid    ${result}


# ---------------------------------------------------------------------------
# calculate_portfolio_returns
# ---------------------------------------------------------------------------

Portfolio Returns Series Is Non-Empty
    [Documentation]    calculate_portfolio_returns on a valid portfolio DataFrame
    ...    returns a non-empty pl.Series.
    [Tags]    utils_data    returns    smoke
    ${returns}=    Calculate Returns From Valid Portfolio Df
    Portfolio Returns Series Should Be Valid    ${returns}
