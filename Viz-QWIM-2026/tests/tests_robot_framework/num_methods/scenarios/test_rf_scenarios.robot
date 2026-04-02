*** Settings ***
Documentation
...    Robot Framework test suite for num_methods.scenarios.
...
...    Covers Scenarios_Distribution (Normal, Student-t, Lognormal),
...    Scenarios_CMA, and Scenarios_Base utility methods:
...    - Construction without error
...    - Generated DataFrame shape
...    - Null-free outputs
...    - Reproducibility (same seed → same output)
...    - Lognormal positivity
...    - CMA annualisation helpers
...    - validate_scenarios, component and matrix accessors
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-02-28

Library     ${CURDIR}${/}keywords_scenarios.py
Library     Collections

Suite Setup      Log    Starting Scenarios Robot Framework Test Suite
Suite Teardown   Log    Scenarios Test Suite Complete


*** Variables ***
${NUM_COMPONENTS}    ${3}
${NUM_DAYS}          ${30}
${SEED_A}            ${42}
${SEED_B}            ${99}
${COMP_0}            US_Equity
${CMA_COMP_0}        US Large Cap


*** Test Cases ***

# ---------------------------------------------------------------------------
# Normal distribution — construction & generation
# ---------------------------------------------------------------------------

Normal Scenario Created Without Error
    [Documentation]    Scenarios_Distribution(NORMAL) constructs without raising.
    [Tags]    scenarios    construction    smoke    normal
    ${sc}=    Create Normal Distribution Scenario
    Should Not Be Equal    ${sc}    ${None}

Normal Scenario Generates Non-Empty DataFrame
    [Documentation]    generate() returns a DataFrame with at least one row.
    [Tags]    scenarios    generation    normal
    ${sc}=    Create Normal Distribution Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    ${rows}=    Get Generated Row Count    ${df}
    Should Be True    ${rows} > 0

Normal Scenario Generated Row Count Matches Num Days
    [Documentation]    Row count equals the requested num_days.
    [Tags]    scenarios    generation    normal
    ${sc}=    Create Normal Distribution Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    ${rows}=    Get Generated Row Count    ${df}
    Should Be Equal As Integers    ${rows}    ${NUM_DAYS}

Normal Scenario Has No Null Values
    [Documentation]    Generated DataFrame must be fully populated.
    [Tags]    scenarios    generation    normal
    ${sc}=    Create Normal Distribution Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    ${ok}=    Dataframe Has No Nulls    ${df}
    Should Be True    ${ok}

Normal Scenario Has Date Column
    [Documentation]    First column of generated DataFrame is 'Date'.
    [Tags]    scenarios    generation    normal
    ${sc}=    Create Normal Distribution Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    ${cols}=    Get Generated Column Names    ${df}
    Should Be Equal    ${cols}[0]    Date


# ---------------------------------------------------------------------------
# Reproducibility — Normal
# ---------------------------------------------------------------------------

Same Seed Produces Identical Normal Scenario
    [Documentation]    Two generates with identical seed yield identical returns.
    [Tags]    scenarios    reproducibility    normal
    ${sc_a}=    Create Normal Distribution Scenario    seed=${SEED_A}
    ${sc_b}=    Create Normal Distribution Scenario    seed=${SEED_A}
    ${df_a}=    Generate Scenario    ${sc_a}
    ${df_b}=    Generate Scenario    ${sc_b}
    @{components}=    Create List    US_Equity    Intl_Equity    US_Bond
    ${same}=    Two Scenarios Identical    ${df_a}    ${df_b}    ${components}
    Should Be True    ${same}

Different Seeds Produce Different Normal Scenarios
    [Documentation]    Two generates with different seeds yield different returns.
    [Tags]    scenarios    reproducibility    normal
    ${sc_a}=    Create Normal Distribution Scenario    seed=${SEED_A}
    ${sc_b}=    Create Normal Distribution Scenario    seed=${SEED_B}
    ${df_a}=    Generate Scenario    ${sc_a}
    ${df_b}=    Generate Scenario    ${sc_b}
    ${differ}=    Two Scenarios Differ    ${df_a}    ${df_b}    ${COMP_0}
    Should Be True    ${differ}


# ---------------------------------------------------------------------------
# Student-t distribution
# ---------------------------------------------------------------------------

Student-t Scenario Created Without Error
    [Documentation]    Scenarios_Distribution(STUDENT_T) constructs without raising.
    [Tags]    scenarios    construction    smoke    student_t
    ${sc}=    Create Student T Scenario
    Should Not Be Equal    ${sc}    ${None}

Student-t Scenario Generates Correct Row Count
    [Documentation]    generate() returns the requested number of rows.
    [Tags]    scenarios    generation    student_t
    ${sc}=    Create Student T Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    ${rows}=    Get Generated Row Count    ${df}
    Should Be Equal As Integers    ${rows}    ${NUM_DAYS}

Student-t Scenario Has No Null Values
    [Documentation]    Student-t generate() must be fully populated.
    [Tags]    scenarios    generation    student_t
    ${sc}=    Create Student T Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    ${ok}=    Dataframe Has No Nulls    ${df}
    Should Be True    ${ok}


# ---------------------------------------------------------------------------
# Lognormal distribution
# ---------------------------------------------------------------------------

Lognormal Scenario Created Without Error
    [Documentation]    Scenarios_Distribution(LOGNORMAL) constructs without raising.
    [Tags]    scenarios    construction    smoke    lognormal
    ${sc}=    Create Lognormal Scenario
    Should Not Be Equal    ${sc}    ${None}

Lognormal Scenario All Values Positive
    [Documentation]    Lognormal returns are strictly > 0.
    [Tags]    scenarios    generation    lognormal
    ${sc}=    Create Lognormal Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    @{components}=    Create List    US_Equity    Intl_Equity    US_Bond
    ${positive}=    All Values Positive    ${df}    ${components}
    Should Be True    ${positive}

Lognormal Scenario Has No Null Values
    [Documentation]    Lognormal generate() must be fully populated.
    [Tags]    scenarios    generation    lognormal
    ${sc}=    Create Lognormal Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    ${ok}=    Dataframe Has No Nulls    ${df}
    Should Be True    ${ok}


# ---------------------------------------------------------------------------
# CMA scenario
# ---------------------------------------------------------------------------

CMA Scenario Created Without Error
    [Documentation]    Scenarios_CMA constructs without raising.
    [Tags]    scenarios    construction    smoke    cma
    ${sc}=    Create Cma Scenario
    Should Not Be Equal    ${sc}    ${None}

CMA Scenario Generates Correct Row Count
    [Documentation]    generate() returns the requested number of rows.
    [Tags]    scenarios    generation    cma
    ${sc}=    Create Cma Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    ${rows}=    Get Generated Row Count    ${df}
    Should Be Equal As Integers    ${rows}    ${NUM_DAYS}

CMA Scenario Has No Null Values
    [Documentation]    CMA generate() must be fully populated.
    [Tags]    scenarios    generation    cma
    ${sc}=    Create Cma Scenario    num_days=${NUM_DAYS}
    ${df}=    Generate Scenario    ${sc}
    ${ok}=    Dataframe Has No Nulls    ${df}
    Should Be True    ${ok}

CMA Daily Vols Less Than Annual Vols
    [Documentation]    Daily volatilities must be strictly less than annual
    ...    (daily = annual / sqrt(252)).
    [Tags]    scenarios    cma    annualisation
    ${sc}=    Create Cma Scenario
    ${ok}=    Get Cma Daily Vols Less Than Annual    ${sc}
    Should Be True    ${ok}

CMA Reports Correct Number Of Asset Classes
    [Documentation]    m_num_components equals the number supplied.
    [Tags]    scenarios    cma    properties
    ${sc}=    Create Cma Scenario    num_components=${NUM_COMPONENTS}
    ${count}=    Cma Num Asset Classes    ${sc}
    Should Be Equal As Integers    ${count}    ${NUM_COMPONENTS}


# ---------------------------------------------------------------------------
# Base-class utilities
# ---------------------------------------------------------------------------

Validate Scenarios Passes After Generate
    [Documentation]    validate_scenarios() completes without exception
    ...    after generate() is called.
    [Tags]    scenarios    validation
    ${sc_raw}=    Create Normal Distribution Scenario    num_days=${NUM_DAYS}
    ${sc}=    Generate And Store    ${sc_raw}
    ${ok}=    Validate Scenarios    ${sc}
    Should Be True    ${ok}

Component Series Length Matches Num Days
    [Documentation]    get_component_series() length equals num_days.
    [Tags]    scenarios    accessor
    ${sc}=    Create Normal Distribution Scenario    num_days=${NUM_DAYS}
    ${sc}=    Generate And Store    ${sc}
    ${length}=    Get Component Series Length    ${sc}    ${COMP_0}
    Should Be Equal As Integers    ${length}    ${NUM_DAYS}

Returns Matrix Shape Is Correct
    [Documentation]    get_returns_matrix() shape is (num_days, num_components).
    [Tags]    scenarios    accessor
    ${sc}=    Create Normal Distribution Scenario
    ...    num_components=${NUM_COMPONENTS}    num_days=${NUM_DAYS}
    ${sc}=    Generate And Store    ${sc}
    ${shape}=    Get Returns Matrix Shape    ${sc}
    Should Be Equal As Integers    ${shape}[0]    ${NUM_DAYS}
    Should Be Equal As Integers    ${shape}[1]    ${NUM_COMPONENTS}
