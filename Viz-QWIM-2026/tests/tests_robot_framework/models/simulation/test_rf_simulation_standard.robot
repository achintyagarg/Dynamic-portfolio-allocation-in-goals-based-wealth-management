*** Settings ***
Documentation
...    Robot Framework test suite for Simulation_Standard (Monte Carlo).
...
...    Verifies simulation construction with Normal, Lognormal and Student-T
...    distributions; run() output shape (rows = num_days + 1,
...    cols = num_scenarios + 1); presence and ascending order of the Date
...    column; all portfolio values positive; first-row values equal
...    initial_value; and reproducibility across identical seeds.
...
...    Test Categories:
...    - Simulation construction (Normal / Lognormal / Student-T)
...    - run() output validity (shape, Date column, positivity)
...    - Initial value in first row
...    - Date column sorted ascending
...    - Reproducibility (same seed → identical output)
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-02-28

Library     ${CURDIR}${/}keywords_simulation_standard.py
Library     Collections

Suite Setup      Log    Starting Simulation Standard Robot Framework Test Suite
Suite Teardown   Log    Simulation Standard Test Suite Complete


*** Variables ***
# Component lists
@{TWO_COMPONENTS}       VTI    AGG
@{THREE_COMPONENTS}     VTI    VXUS    AGG

# Simulation sizing — must match _TEST_NUM_DAYS and _TEST_NUM_SCENARIOS in keywords
${NUM_DAYS}             ${30}
${NUM_SCENARIOS}        ${50}
${INITIAL_VALUE}        ${100.0}

# Expected shape: rows = num_days + 1 = 31, cols = num_scenarios + 1 = 51
${EXPECTED_ROWS}        ${31}
${EXPECTED_COLS}        ${51}


*** Test Cases ***

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

Normal Simulation Created Without Error For Two Components
    [Documentation]    Simulation_Standard with NORMAL distribution and two
    ...    components constructs without raising an exception.
    [Tags]    simulation    construction    smoke
    ${sim}=    Create Simulation Normal    @{TWO_COMPONENTS}
    Should Not Be Equal    ${sim}    ${None}

Lognormal Simulation Created Without Error For Two Components
    [Documentation]    Simulation_Standard with LOGNORMAL distribution
    ...    constructs without raising an exception.
    [Tags]    simulation    construction    smoke
    ${sim}=    Create Simulation Lognormal    @{TWO_COMPONENTS}
    Should Not Be Equal    ${sim}    ${None}

Student T Simulation Created Without Error For Two Components
    [Documentation]    Simulation_Standard with STUDENT_T distribution
    ...    constructs without raising an exception.
    [Tags]    simulation    construction    smoke
    ${sim}=    Create Simulation Student T    @{TWO_COMPONENTS}
    Should Not Be Equal    ${sim}    ${None}

Normal Simulation Created For Three Components
    [Documentation]    Simulation_Standard with NORMAL distribution and three
    ...    components constructs without raising an exception.
    [Tags]    simulation    construction
    ${sim}=    Create Simulation Normal    @{THREE_COMPONENTS}
    Should Not Be Equal    ${sim}    ${None}


# ---------------------------------------------------------------------------
# run() Output Validity (Normal distribution)
# ---------------------------------------------------------------------------

Run Normal Simulation Returns Valid DataFrame
    [Documentation]    simulation.run() returns a non-empty pl.DataFrame
    ...    with at least one column.
    [Tags]    simulation    run    smoke
    ${sim}=    Create Simulation Normal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation Results Dataframe Should Be Valid    ${results}

Normal Simulation Results Contain Date Column
    [Documentation]    The DataFrame returned by run() contains a 'Date' column.
    [Tags]    simulation    run    smoke
    ${sim}=    Create Simulation Normal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation Results Should Have Date Column    ${results}

Normal Simulation Results Have Correct Row Count
    [Documentation]    Number of rows equals num_days + 1 (t=0 through t=num_days).
    [Tags]    simulation    run
    ${sim}=    Create Simulation Normal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation Results Row Count Should Equal    ${results}    ${EXPECTED_ROWS}

Normal Simulation Results Have Correct Column Count
    [Documentation]    Number of columns equals num_scenarios + 1 (Date + one per scenario).
    [Tags]    simulation    run
    ${sim}=    Create Simulation Normal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation Results Column Count Should Equal    ${results}    ${EXPECTED_COLS}

Normal Simulation All Portfolio Values Are Positive
    [Documentation]    Every value in every scenario column is strictly positive.
    [Tags]    simulation    run
    ${sim}=    Create Simulation Normal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation All Portfolio Values Should Be Positive    ${results}

Normal Simulation First Row Equals Initial Value
    [Documentation]    All scenario columns start at initial_value (100.0).
    [Tags]    simulation    run    smoke
    ${sim}=    Create Simulation Normal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation First Row Values Should Equal Initial Value    ${results}    ${INITIAL_VALUE}

Normal Simulation Date Column Is Sorted Ascending
    [Documentation]    The Date column is in strictly ascending order.
    [Tags]    simulation    run
    ${sim}=    Create Simulation Normal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation Date Column Is Sorted Ascending    ${results}


# ---------------------------------------------------------------------------
# run() Output Validity (Lognormal distribution)
# ---------------------------------------------------------------------------

Lognormal Simulation Returns Valid DataFrame
    [Documentation]    run() for a LOGNORMAL simulation returns a valid DataFrame.
    [Tags]    simulation    run    lognormal
    ${sim}=    Create Simulation Lognormal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation Results Dataframe Should Be Valid    ${results}

Lognormal Simulation All Portfolio Values Are Positive
    [Documentation]    Lognormal distribution must always yield positive values.
    [Tags]    simulation    run    lognormal
    ${sim}=    Create Simulation Lognormal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation All Portfolio Values Should Be Positive    ${results}

Lognormal Simulation First Row Equals Initial Value
    [Documentation]    Lognormal simulation t=0 row equals initial_value.
    [Tags]    simulation    run    lognormal
    ${sim}=    Create Simulation Lognormal    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation First Row Values Should Equal Initial Value    ${results}    ${INITIAL_VALUE}


# ---------------------------------------------------------------------------
# run() Output Validity (Student-T distribution)
# ---------------------------------------------------------------------------

Student T Simulation Returns Valid DataFrame
    [Documentation]    run() for a STUDENT_T simulation returns a valid DataFrame.
    [Tags]    simulation    run    student_t
    ${sim}=    Create Simulation Student T    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation Results Dataframe Should Be Valid    ${results}

Student T Simulation First Row Equals Initial Value
    [Documentation]    Student-T simulation t=0 row equals initial_value.
    [Tags]    simulation    run    student_t
    ${sim}=    Create Simulation Student T    @{TWO_COMPONENTS}
    ${results}=    Run Simulation    ${sim}
    Simulation First Row Values Should Equal Initial Value    ${results}    ${INITIAL_VALUE}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

Same Seed Produces Identical Results
    [Documentation]    Two Simulation_Standard instances with the same seed
    ...    produce byte-for-byte identical run() output.
    [Tags]    simulation    reproducibility    smoke
    Two Simulations With Same Seed Should Produce Identical Results    @{TWO_COMPONENTS}

Different Seeds Produce Valid But Potentially Different Results
    [Documentation]    Two simulations with different seeds both produce
    ...    valid DataFrames (best-effort divergence check).
    [Tags]    simulation    reproducibility
    Two Simulations With Different Seeds May Differ    @{TWO_COMPONENTS}
