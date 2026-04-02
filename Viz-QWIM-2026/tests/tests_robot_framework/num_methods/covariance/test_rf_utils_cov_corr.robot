*** Settings ***
Documentation
...    Robot Framework test suite for num_methods.covariance.
...
...    Verifies covariance_matrix construction with multiple estimators,
...    shape and property assertions, validation pass/fail, correlation
...    matrix properties, component accessors, and distance_estimator_type
...    classmethods.
...
...    Test Categories:
...    - Covariance matrix construction (Empirical, LedoitWolf, OAS)
...    - Shape, component count, observation count
...    - validate_covariance_matrix
...    - Diagonal positivity and symmetry
...    - Correlation matrix: diagonal==1, bounds, symmetry
...    - Component variance and covariance accessors
...    - distance_estimator_type classmethods
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-02-28

Library     ${CURDIR}${/}keywords_utils_cov_corr.py
Library     Collections

Suite Setup      Log    Starting Covariance Robot Framework Test Suite
Suite Teardown   Log    Covariance Test Suite Complete


*** Variables ***
${NUM_COMPONENTS}       ${3}
${NUM_OBS}              ${60}
${COMPONENT_AAPL}       AAPL
${COMPONENT_MSFT}       MSFT
${SHAPE_ROW}            ${3}
${SHAPE_COL}            ${3}


*** Test Cases ***

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

Empirical Covariance Matrix Created Without Error
    [Documentation]    Empirical covariance_matrix constructed from synthetic
    ...    data does not raise an exception.
    [Tags]    covariance    construction    smoke
    ${cov}=    Build Empirical Covariance Matrix    num_components=${NUM_COMPONENTS}    num_obs=${NUM_OBS}
    Should Not Be Equal    ${cov}    ${None}

LedoitWolf Covariance Matrix Created Without Error
    [Documentation]    LEDOIT_WOLF estimator constructs without error.
    [Tags]    covariance    construction    smoke
    ${cov}=    Build Covariance Matrix With Estimator    LEDOIT_WOLF
    Should Not Be Equal    ${cov}    ${None}

OAS Covariance Matrix Created Without Error
    [Documentation]    ORACLE_APPROXIMATING_SHRINKAGE estimator constructs without error.
    [Tags]    covariance    construction    smoke
    ${cov}=    Build Covariance Matrix With Estimator    ORACLE_APPROXIMATING_SHRINKAGE
    Should Not Be Equal    ${cov}    ${None}

Shrunk Covariance Matrix Created Without Error
    [Documentation]    SHRUNK_COVARIANCE estimator constructs without error.
    [Tags]    covariance    construction
    ${cov}=    Build Covariance Matrix With Estimator    SHRUNK_COVARIANCE
    Should Not Be Equal    ${cov}    ${None}


# ---------------------------------------------------------------------------
# Shape and properties
# ---------------------------------------------------------------------------

Covariance Matrix Has Correct Shape
    [Documentation]    get_covariance_matrix() returns a (3, 3) array for
    ...    3-component input.
    [Tags]    covariance    properties
    ${cov}=    Build Empirical Covariance Matrix    num_components=${SHAPE_ROW}    num_obs=${NUM_OBS}
    ${shape}=    Get Covariance Matrix Shape    ${cov}
    Should Be Equal As Integers    ${shape}[0]    ${SHAPE_ROW}
    Should Be Equal As Integers    ${shape}[1]    ${SHAPE_COL}

Covariance Matrix Reports Correct Component Count
    [Documentation]    m_num_components equals the number of return columns.
    [Tags]    covariance    properties
    ${cov}=    Build Empirical Covariance Matrix    num_components=${NUM_COMPONENTS}    num_obs=${NUM_OBS}
    ${count}=    Get Num Components    ${cov}
    Should Be Equal As Integers    ${count}    ${NUM_COMPONENTS}

Covariance Matrix Reports Correct Observation Count
    [Documentation]    m_num_observations equals the number of DataFrame rows.
    [Tags]    covariance    properties
    ${cov}=    Build Empirical Covariance Matrix    num_components=${NUM_COMPONENTS}    num_obs=${NUM_OBS}
    ${obs}=    Get Num Observations    ${cov}
    Should Be Equal As Integers    ${obs}    ${NUM_OBS}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

Validate Covariance Matrix Passes Without Error
    [Documentation]    validate_covariance_matrix() completes without raising.
    [Tags]    covariance    validation
    ${cov}=    Build Empirical Covariance Matrix
    ${result}=    Validate Covariance Matrix    ${cov}
    Should Be True    ${result}

Covariance Diagonal Is Positive
    [Documentation]    All diagonal elements (variances) must be > 0.
    [Tags]    covariance    validation
    ${cov}=    Build Empirical Covariance Matrix
    ${result}=    Covariance Diagonal Is Positive    ${cov}
    Should Be True    ${result}

Covariance Matrix Is Symmetric
    [Documentation]    max|Cov - Cov^T| < 1e-8.
    [Tags]    covariance    validation
    ${cov}=    Build Empirical Covariance Matrix
    ${result}=    Covariance Is Symmetric    ${cov}
    Should Be True    ${result}


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

Correlation Matrix Diagonal Equals One
    [Documentation]    All diagonal elements of the correlation matrix must be 1.0.
    [Tags]    covariance    correlation
    ${cov}=    Build Empirical Covariance Matrix
    ${result}=    Correlation Diagonal Equals One    ${cov}
    Should Be True    ${result}

Correlation Values In Bounds
    [Documentation]    All correlation values must lie in [-1, 1].
    [Tags]    covariance    correlation
    ${cov}=    Build Empirical Covariance Matrix
    ${result}=    Correlation Values In Bounds    ${cov}
    Should Be True    ${result}

Correlation Matrix Is Symmetric
    [Documentation]    Correlation matrix is symmetric within 1e-10.
    [Tags]    covariance    correlation
    ${cov}=    Build Empirical Covariance Matrix
    ${result}=    Correlation Is Symmetric    ${cov}
    Should Be True    ${result}


# ---------------------------------------------------------------------------
# Component accessors
# ---------------------------------------------------------------------------

Component Variance Is Positive
    [Documentation]    get_component_variance('AAPL') returns a positive value.
    [Tags]    covariance    accessor
    ${cov}=    Build Empirical Covariance Matrix
    ${var}=    Get Component Variance    ${cov}    ${COMPONENT_AAPL}
    Should Be True    ${var} > 0

Covariance Accessor Symmetric
    [Documentation]    Cov(AAPL, MSFT) == Cov(MSFT, AAPL).
    [Tags]    covariance    accessor
    ${cov}=    Build Empirical Covariance Matrix
    ${ab}=    Get Component Covariance    ${cov}    ${COMPONENT_AAPL}    ${COMPONENT_MSFT}
    ${ba}=    Get Component Covariance    ${cov}    ${COMPONENT_MSFT}    ${COMPONENT_AAPL}
    Should Be Equal As Numbers    ${ab}    ${ba}


# ---------------------------------------------------------------------------
# distance_estimator_type
# ---------------------------------------------------------------------------

Correlation Based Estimators Returns Non-Empty List
    [Documentation]    get_correlation_based() returns at least one member.
    [Tags]    covariance    distance_estimator
    ${result}=    Get Correlation Based Estimators
    ${length}=    Get Length    ${result}
    Should Be True    ${length} >= 1

Rank Based Estimators Returns Non-Empty List
    [Documentation]    get_rank_based() returns at least one member.
    [Tags]    covariance    distance_estimator
    ${result}=    Get Rank Based Estimators
    ${length}=    Get Length    ${result}
    Should Be True    ${length} >= 1

All Distance Estimator Members Are Categorised
    [Documentation]    All enum members appear in at least one category.
    [Tags]    covariance    distance_estimator
    ${result}=    All Distance Estimators Categorised
    Should Be True    ${result}
