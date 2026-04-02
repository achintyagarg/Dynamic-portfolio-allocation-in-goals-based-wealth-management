*** Settings ***
Documentation
...    Robot Framework test suite for daycount (day-count conventions) utilities.
...
...    Verifies year-fraction calculations for all five standard day-count
...    conventions (30/360, 30/365, ACT/360, ACT/365, ACT/ACT) and
...    the multi-period additivity invariant.
...
...    Test Categories:
...    - 30/360 year-fraction smoke and spot checks
...    - 30/365 year-fraction spot check
...    - ACT/360 year-fraction smoke and spot checks
...    - ACT/365 year-fraction smoke and spot checks
...    - ACT/ACT year-fraction smoke and spot checks
...    - Daycount_Convention enum count
...    - Multi-period additivity
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-03-01

Library     ${CURDIR}${/}keywords_daycount.py
Library     Collections

Suite Setup      Log    Starting Daycount Robot Framework Test Suite
Suite Teardown   Log    Daycount Test Suite Complete


*** Variables ***
${HALF_YEAR}            ${0.5}
${FULL_YEAR}            ${1.0}
${CONVENTION_COUNT}     ${5}


*** Test Cases ***

# ---------------------------------------------------------------------------
# Daycount_Convention enum
# ---------------------------------------------------------------------------

Daycount Convention Enum Has Exactly Five Members
    [Documentation]    Daycount_Convention must expose exactly 5 members.
    [Tags]    daycount    smoke    enum
    Daycount Convention Count Should Equal    ${CONVENTION_COUNT}

# ---------------------------------------------------------------------------
# 30/360
# ---------------------------------------------------------------------------

Thirty 360 Six Month Period Is Approximately Half Year
    [Documentation]    30/360 from 2024-01-01 to 2024-07-01 ≈ 0.5.
    [Tags]    daycount    smoke    thirty_360
    ${yf}=    Calculate Year Fraction Thirty 360    2024-01-01    2024-07-01
    Year Fraction Should Be Approximately    ${yf}    ${HALF_YEAR}

Thirty 360 Full Calendar Year Equals One
    [Documentation]    30/360 from 2024-01-01 to 2025-01-01 = 1.0.
    [Tags]    daycount    thirty_360
    ${yf}=    Calculate Year Fraction Thirty 360    2024-01-01    2025-01-01
    Year Fraction Should Be Approximately    ${yf}    ${FULL_YEAR}

Thirty 360 One Month Approximately 1 Twelfth
    [Documentation]    30/360 from 2024-01-01 to 2024-02-01 ≈ 0.08333.
    [Tags]    daycount    thirty_360
    ${yf}=    Calculate Year Fraction Thirty 360    2024-01-01    2024-02-01
    Year Fraction Should Be Approximately    ${yf}    ${0.08333}

# ---------------------------------------------------------------------------
# 30/365
# ---------------------------------------------------------------------------

Thirty 365 Full Calendar Year Approximately 0 98630
    [Documentation]    30/365 from 2024-01-01 to 2025-01-01 ≈ 0.98630.
    [Tags]    daycount    thirty_365
    ${yf}=    Calculate Year Fraction Thirty 365    2024-01-01    2025-01-01
    Year Fraction Should Be Approximately    ${yf}    ${0.98630}

# ---------------------------------------------------------------------------
# ACT/360
# ---------------------------------------------------------------------------

Act 360 Ninety Day Period Greater Than Quarter Year
    [Documentation]    ACT/360 from 2024-01-01 to 2024-04-01 > 0.25.
    [Tags]    daycount    smoke    act_360
    ${yf}=    Calculate Year Fraction Actual 360    2024-01-01    2024-04-01
    Year Fraction Should Be Greater Than    ${yf}    ${0.25}

Act 360 Ninety Days Approximately 0 25278
    [Documentation]    ACT/360 from 2024-01-01 to 2024-04-01 ≈ 0.25278.
    [Tags]    daycount    act_360
    ${yf}=    Calculate Year Fraction Actual 360    2024-01-01    2024-04-01
    Year Fraction Should Be Approximately    ${yf}    ${0.25278}

# ---------------------------------------------------------------------------
# ACT/365
# ---------------------------------------------------------------------------

Act 365 Six Month Period Less Than Half Year
    [Documentation]    ACT/365 from 2024-01-01 to 2024-07-01 < 0.5
    ...    because 2024 is a leap year (181 days / 365 < 0.5).
    [Tags]    daycount    smoke    act_365
    ${yf}=    Calculate Year Fraction Actual 365    2024-01-01    2024-07-01
    Year Fraction Should Be Less Than    ${yf}    ${HALF_YEAR}

Act 365 H1 2024 Approximately 0 49589
    [Documentation]    ACT/365 from 2024-01-01 to 2024-07-01 ≈ 0.49589.
    [Tags]    daycount    act_365
    ${yf}=    Calculate Year Fraction Actual 365    2024-01-01    2024-07-01
    Year Fraction Should Be Approximately    ${yf}    ${0.49589}

# ---------------------------------------------------------------------------
# ACT/ACT
# ---------------------------------------------------------------------------

Act Act Non Leap Year Equals One
    [Documentation]    ACT/ACT from 2023-01-01 to 2024-01-01 = 1.0.
    [Tags]    daycount    smoke    act_act
    ${yf}=    Calculate Year Fraction Actual Actual    2023-01-01    2024-01-01
    Year Fraction Should Be Approximately    ${yf}    ${FULL_YEAR}

Act Act Leap Year Equals One
    [Documentation]    ACT/ACT from 2024-01-01 to 2025-01-01 = 1.0.
    [Tags]    daycount    act_act
    ${yf}=    Calculate Year Fraction Actual Actual    2024-01-01    2025-01-01
    Year Fraction Should Be Approximately    ${yf}    ${FULL_YEAR}

Act Act Half Year On Non Leap Year Approximately 0 5
    [Documentation]    ACT/ACT from 2023-01-01 to 2023-07-02 ≈ 0.5.
    [Tags]    daycount    act_act
    ${yf}=    Calculate Year Fraction Actual Actual    2023-01-01    2023-07-02
    Year Fraction Should Be Approximately    ${yf}    ${HALF_YEAR}

# ---------------------------------------------------------------------------
# Multi-period additivity
# ---------------------------------------------------------------------------

Act Act Two Half Years Sum To Full Year
    [Documentation]    The two ACT/ACT half-year fractions across 2023 must
    ...    sum to approximately 1.0 (additivity invariant).
    [Tags]    daycount    additivity
    ${yf_a}=    Calculate Year Fraction Actual Actual    2023-01-01    2023-07-02
    ${yf_b}=    Calculate Year Fraction Actual Actual    2023-07-02    2024-01-01
    Two Year Fractions Should Sum To Approximately    ${yf_a}    ${yf_b}    ${FULL_YEAR}

Thirty 360 Two Half Years Sum To Full Year
    [Documentation]    The two 30/360 half-year fractions for 2024 sum to 1.0.
    [Tags]    daycount    additivity    thirty_360
    ${yf_a}=    Calculate Year Fraction Thirty 360    2024-01-01    2024-07-01
    ${yf_b}=    Calculate Year Fraction Thirty 360    2024-07-01    2025-01-01
    Two Year Fractions Should Sum To Approximately    ${yf_a}    ${yf_b}    ${FULL_YEAR}
