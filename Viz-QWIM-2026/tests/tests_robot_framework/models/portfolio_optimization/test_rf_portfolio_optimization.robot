*** Settings ***
Documentation
...    Robot Framework test suite for portfolio optimization.
...
...    Covers three areas:
...
...    1. Enum smoke tests — portfolio_optimization_type and
...       portfolio_optimization_feature_type member counts and classmethod counts.
...       (Regression guard for the aenum → stdlib enum migration.)
...
...    2. Bug-fix regression — benchmark_pandas possibly-unbound variable fix:
...       convex optimizations that do NOT use CONVEX_BENCHMARK_TRACKING must
...       run without UnboundLocalError.
...
...    3. End-to-end output validity — skfolio and azapy wrappers must produce
...       portfolio_QWIM objects with positive, normalized weights.
...
...    Test Categories (tags):
...     @portfolio_optimization — all tests in this suite
...     @enum                  — enum structure tests
...     @regression            — bug-fix and migration regression tests
...     @e2e                   — end-to-end optimization tests
...     @smoke                 — fast sanity checks
...
...    Author:         QWIM Development Team
...    Version:        0.1.0
...    Last Modified:  2027-01-01

Library     ${CURDIR}${/}keywords_portfolio_optimization.py
Library     Collections

Suite Setup      Log    Starting Portfolio Optimization Robot Framework Test Suite
Suite Teardown   Log    Portfolio Optimization Test Suite Complete


*** Variables ***

# Expected counts (pinned; tests fail if enum structure changes accidentally)
${EXPECTED_OPT_TYPE_MEMBERS}        ${13}
${EXPECTED_FEATURE_TYPE_MEMBERS}    ${24}

${EXPECTED_BASIC_METHODS}           ${3}
${EXPECTED_CONVEX_METHODS}          ${5}
${EXPECTED_CLUSTERING_METHODS}      ${4}
${EXPECTED_ENSEMBLE_METHODS}        ${1}

${EXPECTED_OBJECTIVES}              ${4}
${EXPECTED_ALL_CONSTRAINTS}         ${12}
${EXPECTED_CONVEX_FEATURES}         ${13}


*** Test Cases ***

# ===========================================================================
# portfolio_optimization_type — enum structure
# ===========================================================================

Optimization Type Enum Has 13 Members
    [Documentation]    Regression guard: adding/removing a member breaks this test.
    [Tags]    portfolio_optimization    enum    smoke
    ${count}=    Get Optimization Type Member Count
    Should Be Equal As Integers    ${count}    ${EXPECTED_OPT_TYPE_MEMBERS}

Optimization Type Values Equal Names
    [Documentation]    Each enum value string must equal its member name.
    [Tags]    portfolio_optimization    enum    smoke
    ${result}=    Enum Value Equals Name For All Opt Type Members
    Should Be True    ${result}

Get Basic Methods Returns 3 Members
    [Documentation]    get_basic_methods must return exactly 3 items.
    [Tags]    portfolio_optimization    enum    smoke
    ${count}=    Get Basic Methods Count
    Should Be Equal As Integers    ${count}    ${EXPECTED_BASIC_METHODS}

Get Convex Methods Returns 5 Members
    [Documentation]    get_convex_methods must return exactly 5 items.
    [Tags]    portfolio_optimization    enum    smoke
    ${count}=    Get Convex Methods Count
    Should Be Equal As Integers    ${count}    ${EXPECTED_CONVEX_METHODS}

Get Clustering Methods Returns 4 Members
    [Documentation]    get_clustering_methods must return exactly 4 items.
    [Tags]    portfolio_optimization    enum    smoke
    ${count}=    Get Clustering Methods Count
    Should Be Equal As Integers    ${count}    ${EXPECTED_CLUSTERING_METHODS}

Get Ensemble Methods Returns 1 Member
    [Documentation]    get_ensemble_methods must return exactly 1 item.
    [Tags]    portfolio_optimization    enum    smoke
    ${count}=    Get Ensemble Methods Count
    Should Be Equal As Integers    ${count}    ${EXPECTED_ENSEMBLE_METHODS}

All Method Groups Cover All 13 Enum Members
    [Documentation]    Union of basic+convex+clustering+ensemble must be the full enum.
    [Tags]    portfolio_optimization    enum    regression
    ${result}=    All Method Groups Cover All Members
    Should Be True    ${result}

Subscript Lookup Works After Enum Migration
    [Documentation]    portfolio_optimization_type['BASIC_EQUAL_WEIGHTED'] must work.
    ...    This was broken when aenum.Enum was used — stdlib Enum subscript
    ...    behaves differently.
    [Tags]    portfolio_optimization    enum    regression
    ${result}=    Subscript Lookup Works
    Should Be True    ${result}


# ===========================================================================
# portfolio_optimization_feature_type — enum structure
# ===========================================================================

Feature Type Enum Has 24 Members
    [Documentation]    Regression guard for feature enum member count.
    [Tags]    portfolio_optimization    enum    smoke
    ${count}=    Get Feature Type Member Count
    Should Be Equal As Integers    ${count}    ${EXPECTED_FEATURE_TYPE_MEMBERS}

Feature Type Values Equal Names
    [Documentation]    Each feature enum value must equal its member name.
    [Tags]    portfolio_optimization    enum    smoke
    ${result}=    Enum Value Equals Name For All Feature Type Members
    Should Be True    ${result}

Get Objectives Returns 4 Features
    [Documentation]    get_objectives must return exactly 4 objective features.
    [Tags]    portfolio_optimization    enum    smoke
    ${count}=    Get Objectives Count
    Should Be Equal As Integers    ${count}    ${EXPECTED_OBJECTIVES}

Get All Constraints Returns 12 Features
    [Documentation]    get_all_constraints must return 12 (5 weight + 6 portfolio + 1 custom).
    [Tags]    portfolio_optimization    enum    smoke
    ${count}=    Get All Constraints Count
    Should Be Equal As Integers    ${count}    ${EXPECTED_ALL_CONSTRAINTS}

Get Convex Features Returns 13 Features
    [Documentation]    get_convex_features must return exactly 13 items.
    [Tags]    portfolio_optimization    enum    smoke
    ${count}=    Get Convex Features Count
    Should Be Equal As Integers    ${count}    ${EXPECTED_CONVEX_FEATURES}

Integer Features Are Disjoint From Convex Features
    [Documentation]    Integer/combinatorial features must not overlap with convex features.
    [Tags]    portfolio_optimization    enum    regression
    ${result}=    Integer Features Disjoint From Convex Features
    Should Be True    ${result}


# ===========================================================================
# End-to-end: skfolio optimization → portfolio_QWIM
# ===========================================================================

Equal Weighted 3 Asset Portfolio Is Valid Portfolio QWIM
    [Documentation]    calc_skfolio_optimization_basic with BASIC_EQUAL_WEIGHTED must
    ...    return a portfolio_QWIM with 3 components and weights summing to 1.
    [Tags]    portfolio_optimization    e2e    smoke
    ${port}=    Run Skfolio Basic Equal Weighted 3 Assets
    ${is_qwim}=    Portfolio Is Portfolio QWIM    ${port}
    Should Be True    ${is_qwim}
    ${n}=    Portfolio Num Components    ${port}
    Should Be Equal As Integers    ${n}    ${3}

Equal Weighted 3 Asset Portfolio Weights Sum To One
    [Documentation]    Equal-weighted portfolio weights must sum to 1.0 ± 1e-5.
    [Tags]    portfolio_optimization    e2e    smoke
    ${port}=    Run Skfolio Basic Equal Weighted 3 Assets
    ${weight_sum}=    Portfolio Weights Sum    ${port}
    Should Be True    abs(${weight_sum} - 1.0) < 1e-5

Equal Weighted Weights Are Correct
    [Documentation]    Each weight in an equal-weighted 3-asset portfolio must be 1/3.
    [Tags]    portfolio_optimization    e2e    smoke
    ${port}=    Run Skfolio Basic Equal Weighted 3 Assets
    ${ok}=    Equal Weighted Weights Correct    ${port}
    Should Be True    ${ok}

Inverse Volatility 5 Asset Portfolio Is Valid
    [Documentation]    BASIC_INVERSE_VOLATILITY on 5 assets must produce valid portfolio_QWIM.
    [Tags]    portfolio_optimization    e2e    smoke
    ${port}=    Run Skfolio Basic Inverse Vol 5 Assets
    ${is_qwim}=    Portfolio Is Portfolio QWIM    ${port}
    Should Be True    ${is_qwim}
    ${n}=    Portfolio Num Components    ${port}
    Should Be Equal As Integers    ${n}    ${5}

Inverse Volatility All Weights Non Negative
    [Documentation]    Inverse-volatility weights must all be >= 0.
    [Tags]    portfolio_optimization    e2e    smoke
    ${port}=    Run Skfolio Basic Inverse Vol 5 Assets
    ${ok}=    All Weights Non Negative    ${port}
    Should Be True    ${ok}


# ===========================================================================
# Bug-fix regression: benchmark_pandas unbound variable
# ===========================================================================

Convex Risk Budgeting Runs Without Error
    [Documentation]    CONVEX_RISK_BUDGETING must not raise UnboundLocalError.
    ...    Before the benchmark_pandas fix, any non-BENCHMARK_TRACKING convex
    ...    call would fail because benchmark_pandas was only assigned inside an
    ...    elif block.
    [Tags]    portfolio_optimization    e2e    regression
    ${port}=    Run Skfolio Convex Risk Budgeting 5 Assets
    ${is_qwim}=    Portfolio Is Portfolio QWIM    ${port}
    Should Be True    ${is_qwim}

Convex Risk Budgeting Weights Sum To One
    [Documentation]    Risk-budgeting (risk-parity) weights must sum to 1.0.
    [Tags]    portfolio_optimization    e2e    regression
    ${port}=    Run Skfolio Convex Risk Budgeting 5 Assets
    ${weight_sum}=    Portfolio Weights Sum    ${port}
    Should Be True    abs(${weight_sum} - 1.0) < 1e-5

HRP Clustering 5 Asset Portfolio Is Valid
    [Documentation]    HRP clustering must produce a valid portfolio_QWIM with 5 components.
    [Tags]    portfolio_optimization    e2e    smoke
    ${port}=    Run Skfolio Clustering HRP 5 Assets
    ${is_qwim}=    Portfolio Is Portfolio QWIM    ${port}
    Should Be True    ${is_qwim}
    ${n}=    Portfolio Num Components    ${port}
    Should Be Equal As Integers    ${n}    ${5}


# ===========================================================================
# End-to-end: azapy wrapper cross-check
# ===========================================================================

Azapy Inverse Volatility 5 Asset Portfolio Is Valid
    [Documentation]    azapy inverse-volatility must produce a valid portfolio_QWIM.
    [Tags]    portfolio_optimization    e2e    smoke
    ${port}=    Run Azapy Inverse Volatility 5 Assets
    ${is_qwim}=    Portfolio Is Portfolio QWIM    ${port}
    Should Be True    ${is_qwim}
    ${n}=    Portfolio Num Components    ${port}
    Should Be Equal As Integers    ${n}    ${5}

Azapy Inverse Volatility Weights Sum To One
    [Documentation]    azapy inverse-volatility weights must sum to 1.0.
    [Tags]    portfolio_optimization    e2e    smoke
    ${port}=    Run Azapy Inverse Volatility 5 Assets
    ${weight_sum}=    Portfolio Weights Sum    ${port}
    Should Be True    abs(${weight_sum} - 1.0) < 1e-5

Azapy Inverse Volatility All Weights Non Negative
    [Documentation]    azapy inverse-volatility must produce non-negative weights.
    [Tags]    portfolio_optimization    e2e    smoke
    ${port}=    Run Azapy Inverse Volatility 5 Assets
    ${ok}=    All Weights Non Negative    ${port}
    Should Be True    ${ok}
