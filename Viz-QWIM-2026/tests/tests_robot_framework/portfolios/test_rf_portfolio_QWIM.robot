*** Settings ***
Documentation
...    Robot Framework test suite for portfolio_QWIM class and utils_portfolio.
...
...    Verifies portfolio construction from component names and from CSV
...    weights, property access (name, component count, component list,
...    weights DataFrame), weight-sum validation, portfolio value
...    time-series calculation, and benchmark portfolio creation.
...
...    Test Categories:
...    - Portfolio construction from component names (equal weights)
...    - Portfolio construction from CSV weights file
...    - Portfolio property access (name, count, components, weights)
...    - Weight normalisation and sum validation
...    - Portfolio value time-series calculation
...    - Benchmark portfolio creation and divergence check
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-02-28

Library     ${CURDIR}${/}keywords_portfolio_QWIM.py
Library     Collections

Suite Setup      Log    Starting Portfolio QWIM Robot Framework Test Suite
Suite Teardown   Log    Portfolio QWIM Test Suite Complete


*** Variables ***
${PORTFOLIO_NAME}           Tech Portfolio
${CSV_PORTFOLIO_NAME}       CSV Portfolio
${INITIAL_VALUE}            ${100.0}
@{THREE_COMPONENTS}         VTI    AGG    VNQ
@{FIVE_COMPONENTS}          VTI    VXUS    AGG    VNQ    GLD


*** Test Cases ***

# ---------------------------------------------------------------------------
# Portfolio Construction — from component names
# ---------------------------------------------------------------------------

Portfolio Created From Three Component Names Has Correct Name
    [Documentation]    A portfolio constructed from 3 component names stores
    ...    the exact name supplied at construction time.
    [Tags]    portfolio    construction    smoke
    ${portfolio}=    Create Portfolio From Component Names    ${PORTFOLIO_NAME}    @{THREE_COMPONENTS}
    Portfolio Name Should Equal    ${portfolio}    ${PORTFOLIO_NAME}

Portfolio Created From Three Component Names Has Correct Count
    [Documentation]    A portfolio constructed from 3 component names reports
    ...    get_num_components == 3.
    [Tags]    portfolio    construction    smoke
    ${portfolio}=    Create Portfolio From Component Names    ${PORTFOLIO_NAME}    @{THREE_COMPONENTS}
    Portfolio Num Components Should Equal    ${portfolio}    3

Portfolio Created From Two Components Contains Both Components
    [Documentation]    get_portfolio_components list includes every component
    ...    name supplied at construction.
    [Tags]    portfolio    construction
    ${portfolio}=    Create Portfolio From Component Names    Two Asset    VTI    BND
    Portfolio Should Contain Component    ${portfolio}    VTI
    Portfolio Should Contain Component    ${portfolio}    BND

Portfolio Created From Five Component Names Has Count Five
    [Documentation]    Portfolio built from 5 components reports the correct count.
    [Tags]    portfolio    construction
    ${portfolio}=    Create Portfolio From Component Names    Five Assets    @{FIVE_COMPONENTS}
    Portfolio Num Components Should Equal    ${portfolio}    5

Portfolio Created From CSV Weights Has Non-Empty Weights DataFrame
    [Documentation]    A portfolio initialised from the CSV weights file has a
    ...    valid non-empty weights DataFrame.
    [Tags]    portfolio    construction    csv
    ${portfolio}=    Create Portfolio From Weights Csv    ${CSV_PORTFOLIO_NAME}
    Portfolio Weights Dataframe Should Be Valid    ${portfolio}


# ---------------------------------------------------------------------------
# Portfolio Property Access
# ---------------------------------------------------------------------------

Get Portfolio Name Returns The Correct String
    [Documentation]    get_portfolio_name returns exactly the name set at
    ...    construction time.
    [Tags]    portfolio    properties
    ${portfolio}=    Create Portfolio From Component Names    Exact Name    VTI    AGG
    ${name}=    Get Portfolio Name    ${portfolio}
    Should Be Equal    ${name}    Exact Name

Get Portfolio Num Components Returns Correct Integer
    [Documentation]    get_num_components returns an integer matching the
    ...    number of components supplied.
    [Tags]    portfolio    properties
    ${portfolio}=    Create Portfolio From Component Names    Count Test    VTI    AGG    BND
    ${count}=    Get Portfolio Num Components    ${portfolio}
    Should Be Equal As Numbers    ${count}    3

Get Portfolio Components List Contains All Input Names
    [Documentation]    get_portfolio_components returns a list that contains
    ...    every component name provided at construction.
    [Tags]    portfolio    properties
    ${portfolio}=    Create Portfolio From Component Names    List Test    VTI    VXUS    BND
    ${components}=    Get Portfolio Components List    ${portfolio}
    Should Contain    ${components}    VTI
    Should Contain    ${components}    VXUS
    Should Contain    ${components}    BND

Get Portfolio Weights DataFrame Is Non-Empty Polars DataFrame
    [Documentation]    get_portfolio_weights() returns a non-empty pl.DataFrame.
    [Tags]    portfolio    properties
    ${portfolio}=    Create Portfolio From Component Names    Weights DF Test    VTI    AGG
    Portfolio Weights Dataframe Should Be Valid    ${portfolio}


# ---------------------------------------------------------------------------
# Weight Validation
# ---------------------------------------------------------------------------

Equal Weights For Two Components Sum To One
    [Documentation]    When constructing from component names, auto-assigned
    ...    equal weights sum to 1.0 per row.
    [Tags]    portfolio    weights    validation    smoke
    ${portfolio}=    Create Portfolio From Component Names    Sum2    VTI    AGG
    Portfolio Weights Should Sum To One    ${portfolio}

Equal Weights For Four Components Sum To One
    [Documentation]    Auto-assigned equal weights for 4 components sum to 1.0
    ...    (verifies rounding/normalisation).
    [Tags]    portfolio    weights    validation
    ${portfolio}=    Create Portfolio From Component Names    Sum4    VTI    AGG    VNQ    GLD
    Portfolio Weights Should Sum To One    ${portfolio}

Validate And Normalise Portfolio Weights Does Not Raise
    [Documentation]    Calling validate_all_weights(normalize=True) on a valid
    ...    portfolio completes without exceptions.
    [Tags]    portfolio    weights    validation
    ${portfolio}=    Create Portfolio From Component Names    Validate Test    VTI    BND
    Validate And Normalise Portfolio Weights    ${portfolio}

CSV Portfolio Weights Sum To One Per Row
    [Documentation]    Weights loaded from the sample CSV file sum to 1.0 per row
    ...    after portfolio construction.
    [Tags]    portfolio    weights    validation    csv
    ${portfolio}=    Create Portfolio From Weights Csv    CSV Validation
    Portfolio Weights Should Sum To One    ${portfolio}


# ---------------------------------------------------------------------------
# Portfolio Value Calculation
# ---------------------------------------------------------------------------

Calculate Portfolio Values Returns Valid DataFrame
    [Documentation]    calculate_portfolio_values returns a non-empty pl.DataFrame
    ...    with a Portfolio_Value column.
    [Tags]    portfolio    calculation    integration
    ${portfolio}=    Create Portfolio From Weights Csv    Calc Portfolio
    ${values}=    Calculate Portfolio Values From Csv    ${portfolio}    ${INITIAL_VALUE}
    Portfolio Values Dataframe Should Be Valid    ${values}

Portfolio Values First Row Equals Initial Value
    [Documentation]    The first entry of the Portfolio_Value time series equals
    ...    the initial_value of 100.0 supplied to calculate_portfolio_values.
    [Tags]    portfolio    calculation    integration    smoke
    ${portfolio}=    Create Portfolio From Weights Csv    Init Value Portfolio
    ${values}=    Calculate Portfolio Values From Csv    ${portfolio}    ${INITIAL_VALUE}
    Portfolio Values First Row Should Equal    ${values}    ${INITIAL_VALUE}

Portfolio Values Are All Positive
    [Documentation]    Every Portfolio_Value entry in the time series is > 0.
    [Tags]    portfolio    calculation    integration
    ${portfolio}=    Create Portfolio From Weights Csv    Positive Check Portfolio
    ${values}=    Calculate Portfolio Values From Csv    ${portfolio}    ${INITIAL_VALUE}
    Portfolio Values Should All Be Positive    ${values}

Benchmark Portfolio Values Differ From Source Portfolio Values
    [Documentation]    create_benchmark_portfolio_values produces a DataFrame
    ...    with values that diverge from the source portfolio.
    [Tags]    portfolio    benchmark    integration
    ${portfolio}=    Create Portfolio From Weights Csv    Benchmark Source
    ${port_values}=    Calculate Portfolio Values From Csv    ${portfolio}    ${INITIAL_VALUE}
    ${bench_values}=    Create Benchmark From Portfolio Values    ${port_values}
    Benchmark Values Should Differ From Portfolio    ${port_values}    ${bench_values}
