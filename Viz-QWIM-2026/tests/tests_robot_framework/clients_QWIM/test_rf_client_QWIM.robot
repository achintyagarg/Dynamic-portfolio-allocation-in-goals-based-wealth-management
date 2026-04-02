*** Settings ***
Documentation
...    Robot Framework test suite for Client_QWIM class.
...
...    Verifies client construction with primary and partner types,
...    personal information update and retrieval, age and risk-tolerance
...    property access, asset update and total-asset calculation,
...    goal update and DataFrame validation, and income update.
...
...    Test Categories:
...    - Client construction (primary and partner types)
...    - Personal information update and property access
...    - Asset update and total asset calculation
...    - Goal update and goals DataFrame validation
...    - Income update
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-02-28

Library     ${CURDIR}${/}keywords_client_QWIM.py
Library     Collections

Suite Setup      Log    Starting Client QWIM Robot Framework Test Suite
Suite Teardown   Log    Client QWIM Test Suite Complete


*** Variables ***
${PRIMARY_ID}           CLI_P_001
${PARTNER_ID}           CLI_Q_001
${FIRST_NAME}           Jane
${LAST_NAME}            Doe
${CURRENT_AGE}          ${45}
${RETIREMENT_AGE}       ${65}
${RISK_TOLERANCE}       ${6}
${TAXABLE_ASSETS}       ${100000.0}
${TAX_DEFERRED}         ${200000.0}
${TAX_FREE}             ${50000.0}
${TOTAL_ASSETS}         ${350000.0}
${ESSENTIAL_EXPENSE}    ${60000.0}
${IMPORTANT_EXPENSE}    ${20000.0}
${ASPIRATIONAL}         ${10000.0}
${SOCIAL_SECURITY}      ${24000.0}
${PENSION}              ${12000.0}


*** Test Cases ***

# ---------------------------------------------------------------------------
# Client Construction
# ---------------------------------------------------------------------------

Primary Client Created Without Error
    [Documentation]    Client_QWIM constructed with CLIENT_PRIMARY type
    ...    does not raise an exception.
    [Tags]    client    construction    smoke
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Should Not Be Equal    ${client}    ${None}

Partner Client Created Without Error
    [Documentation]    Client_QWIM constructed with CLIENT_PARTNER type
    ...    does not raise an exception.
    [Tags]    client    construction    smoke
    ${client}=    Create Partner Client    ${PARTNER_ID}    John    Doe
    Should Not Be Equal    ${client}    ${None}

Two Distinct Clients Are Different Objects
    [Documentation]    Constructing a primary and a partner client returns
    ...    two independent objects.
    [Tags]    client    construction
    ${primary}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    ${partner}=    Create Partner Client    ${PARTNER_ID}    John    Doe
    Should Not Be Equal    ${primary}    ${partner}


# ---------------------------------------------------------------------------
# Personal Information Update and Access
# ---------------------------------------------------------------------------

Update Personal Info Returns True
    [Documentation]    update_personal_info with valid data returns True.
    [Tags]    client    personal_info    smoke
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Personal Info Should Return True    ${client}    ${FIRST_NAME}    ${LAST_NAME}    ${CURRENT_AGE}    ${RETIREMENT_AGE}    ${RISK_TOLERANCE}

Current Age Stored Correctly After Update
    [Documentation]    get_current_age() returns the age supplied to
    ...    update_personal_info.
    [Tags]    client    personal_info
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Client Personal Info    ${client}    ${FIRST_NAME}    ${LAST_NAME}    ${CURRENT_AGE}    ${RETIREMENT_AGE}    ${RISK_TOLERANCE}
    Client Current Age Should Equal    ${client}    ${CURRENT_AGE}

Retirement Age Stored Correctly After Update
    [Documentation]    get_retirement_age() returns the retirement age supplied
    ...    to update_personal_info.
    [Tags]    client    personal_info
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Client Personal Info    ${client}    ${FIRST_NAME}    ${LAST_NAME}    ${CURRENT_AGE}    ${RETIREMENT_AGE}    ${RISK_TOLERANCE}
    Client Retirement Age Should Equal    ${client}    ${RETIREMENT_AGE}

Risk Tolerance Stored Correctly After Update
    [Documentation]    get_risk_tolerance() returns the value supplied to
    ...    update_personal_info.
    [Tags]    client    personal_info
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Client Personal Info    ${client}    ${FIRST_NAME}    ${LAST_NAME}    ${CURRENT_AGE}    ${RETIREMENT_AGE}    ${RISK_TOLERANCE}
    Client Risk Tolerance Should Equal    ${client}    ${RISK_TOLERANCE}

Personal Info DataFrame Is Non-Empty After Update
    [Documentation]    get_personal_info() returns a non-empty DataFrame after
    ...    update_personal_info has been called.
    [Tags]    client    personal_info
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Client Personal Info    ${client}    ${FIRST_NAME}    ${LAST_NAME}    ${CURRENT_AGE}    ${RETIREMENT_AGE}    ${RISK_TOLERANCE}
    Personal Info Dataframe Should Be Valid    ${client}

Retirement Age Greater Than Current Age After Update
    [Documentation]    Retirement age (65) is greater than current age (45)
    ...    after update_personal_info.
    [Tags]    client    personal_info
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Client Personal Info    ${client}    ${FIRST_NAME}    ${LAST_NAME}    ${CURRENT_AGE}    ${RETIREMENT_AGE}    ${RISK_TOLERANCE}
    ${curr_age}=    Get Client Current Age    ${client}
    ${ret_age}=     Get Client Retirement Age    ${client}
    Should Be True    ${ret_age} > ${curr_age}


# ---------------------------------------------------------------------------
# Asset Update and Calculation
# ---------------------------------------------------------------------------

Update Assets Returns True
    [Documentation]    update_assets with a valid asset list returns True.
    [Tags]    client    assets    smoke
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Assets Should Return True    ${client}

Total Assets Equal Sum Of Individual Categories
    [Documentation]    get_total_assets() equals taxable + tax_deferred + tax_free
    ...    values supplied to update_assets.
    [Tags]    client    assets
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Client Assets    ${client}    ${TAXABLE_ASSETS}    ${TAX_DEFERRED}    ${TAX_FREE}
    Client Total Assets Should Equal    ${client}    ${TOTAL_ASSETS}

Total Assets Are Positive After Update
    [Documentation]    get_total_assets() > 0 after supplying non-zero asset values.
    [Tags]    client    assets
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Client Assets    ${client}    ${TAXABLE_ASSETS}    ${TAX_DEFERRED}    ${TAX_FREE}
    Client Total Assets Should Be Positive    ${client}


# ---------------------------------------------------------------------------
# Goals Update and Validation
# ---------------------------------------------------------------------------

Update Goals Returns True
    [Documentation]    update_goals with a valid goals list returns True.
    [Tags]    client    goals    smoke
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Goals Should Return True    ${client}

Goals DataFrame Is Non-Empty After Update
    [Documentation]    get_goals() returns a non-empty DataFrame after
    ...    update_goals has been called.
    [Tags]    client    goals
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Client Goals    ${client}    ${ESSENTIAL_EXPENSE}    ${IMPORTANT_EXPENSE}    ${ASPIRATIONAL}
    Client Goals Dataframe Should Be Valid    ${client}

Goals Dataframe Is Returned As Polars DataFrame
    [Documentation]    get_goals() returns a pl.DataFrame instance
    ...    (not None and not pandas).
    [Tags]    client    goals
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Client Goals    ${client}    ${ESSENTIAL_EXPENSE}    ${IMPORTANT_EXPENSE}    ${ASPIRATIONAL}
    ${df}=    Get Client Goals Dataframe    ${client}
    Should Not Be Equal    ${df}    ${None}


# ---------------------------------------------------------------------------
# Income Update
# ---------------------------------------------------------------------------

Update Income Returns True With Sample Data
    [Documentation]    update_income with sample social security and pension
    ...    values returns True.
    [Tags]    client    income    smoke
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    Update Income Should Return True    ${client}

Update Income Returns True With Zero Values
    [Documentation]    update_income with zero income values still returns True
    ...    (edge case: client has no fixed income).
    [Tags]    client    income
    ${client}=    Create Primary Client    ${PRIMARY_ID}    ${FIRST_NAME}    ${LAST_NAME}
    ${result}=    Update Client Income    ${client}    ${0.0}    ${0.0}
    Should Be True    ${result}
