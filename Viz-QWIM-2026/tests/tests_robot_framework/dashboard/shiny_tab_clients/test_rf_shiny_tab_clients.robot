*** Settings ***
Documentation     Robot Framework tests for the ``shiny_tab_clients`` dashboard module.
...
...               These tests validate:
...               - Input ID naming conventions
...               - Default values from source code
...               - Business constraint bounds (age, assets, income)
...               - Currency formatting behaviour
...               - Module API surface (callables exposed)
...               - tab_clients orchestration return keys
...
...               No live Shiny server is required; all tests operate on
...               pure-Python module logic via the keyword library.
...
...               Run:
...                   robot --outputdir results/robot_all tests/tests_robot_framework/dashboard/shiny_tab_clients/
...
...               Tags used:
...               ``clients_tab``  – All tests in this suite
...               ``smoke``        – Fast module-import / API surface tests
...               ``personal_info`` / ``assets`` / ``goals`` / ``income`` / ``summary``
...                                – Subtab-specific tests
...               ``naming``       – Input ID convention tests
...               ``defaults``     – Default value tests
...               ``constraints``  – Bound enforcement tests
...               ``currency``     – Currency formatting tests
...               ``integration``  – Cross-module interaction tests
...
...               Author:  QWIM Development Team
...               Version: 0.2.0

Library           ${CURDIR}${/}keywords_shiny_tab_clients.py

Suite Setup       Log    Starting shiny_tab_clients Robot Framework test suite
Suite Teardown    Log    Finished shiny_tab_clients Robot Framework test suite

Force Tags        clients_tab


*** Variables ***
# Subtab short module names (must match src.dashboard.shiny_tab_clients.<name>)
${MOD_PERSONAL_INFO}      subtab_personal_info
${MOD_ASSETS}             subtab_assets
${MOD_GOALS}              subtab_goals
${MOD_INCOME}             subtab_income
${MOD_SUMMARY}            subtab_summary
${MOD_TAB_CLIENTS}        tab_clients

# Expected default values
${DEFAULT_PRIMARY_NAME}            Anne Smith
${DEFAULT_PARTNER_NAME}            William Smith
${DEFAULT_PRIMARY_AGE_CURRENT}     60
${DEFAULT_PRIMARY_AGE_RETIREMENT}  65
${DEFAULT_MARITAL_STATUS}          married
${DEFAULT_RISK_TOLERANCE}          moderate
${DEFAULT_PRIMARY_GENDER}          female
${DEFAULT_PARTNER_GENDER}          male
${DEFAULT_ZIP_CODE}                12345

# Constraint bounds
${AGE_CURRENT_MIN}    18
${AGE_CURRENT_MAX}    100
${AGE_RETIRE_MIN}     50
${AGE_RETIRE_MAX}     80
${ASSET_MAX}          100000000
${INCOME_MAX}         5000000


*** Test Cases ***

# ---------------------------------------------------------------------------
# SMOKE — Module API surface
# ---------------------------------------------------------------------------

All Subtab UI Functions Are Callable
    [Documentation]    Every subtab module exposes a callable UI function.
    [Tags]             smoke    api_surface
    Module Should Expose Callable    ${MOD_PERSONAL_INFO}    subtab_clients_personal_info_ui
    Module Should Expose Callable    ${MOD_ASSETS}           subtab_clients_assets_ui
    Module Should Expose Callable    ${MOD_GOALS}            subtab_clients_goals_ui
    Module Should Expose Callable    ${MOD_INCOME}           subtab_clients_income_ui
    Module Should Expose Callable    ${MOD_SUMMARY}          subtab_clients_summary_ui
    Module Should Expose Callable    ${MOD_TAB_CLIENTS}      tab_clients_ui

All Subtab Server Functions Are Callable
    [Documentation]    Every subtab module exposes a callable server function.
    [Tags]             smoke    api_surface
    Module Should Expose Callable    ${MOD_PERSONAL_INFO}    subtab_clients_personal_info_server
    Module Should Expose Callable    ${MOD_ASSETS}           subtab_clients_assets_server
    Module Should Expose Callable    ${MOD_GOALS}            subtab_clients_goals_server
    Module Should Expose Callable    ${MOD_INCOME}           subtab_clients_income_server
    Module Should Expose Callable    ${MOD_SUMMARY}          subtab_clients_summary_server
    Module Should Expose Callable    ${MOD_TAB_CLIENTS}      tab_clients_server


# ---------------------------------------------------------------------------
# INPUT ID NAMING CONVENTION
# ---------------------------------------------------------------------------

Primary Client Name Input ID Follows Convention
    [Documentation]    Primary name input ID follows hierarchical naming convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_primary_name

Primary Client Age Current Input ID Follows Convention
    [Documentation]    Primary current-age input ID follows hierarchical naming convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_current

Primary Client Age Retirement Input ID Follows Convention
    [Documentation]    Primary retirement-age input ID follows hierarchical naming convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_retirement

Primary Client Age Income Starting Input ID Follows Convention
    [Documentation]    Primary income-start-age input ID follows convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_income_starting

Primary Client Marital Status Input ID Follows Convention
    [Documentation]    Primary marital_status input ID follows convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_primary_status_marital

Primary Client Gender Input ID Follows Convention
    [Documentation]    Primary gender input ID follows convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_primary_gender

Primary Client Risk Tolerance Input ID Follows Convention
    [Documentation]    Primary risk tolerance input ID follows convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_primary_tolerance_risk

Primary Client State Input ID Follows Convention
    [Documentation]    Primary state input ID follows convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_primary_state

Primary Client ZIP Code Input ID Follows Convention
    [Documentation]    Primary ZIP code input ID follows convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_primary_code_zip

Partner Client Name Input ID Follows Convention
    [Documentation]    Partner name input ID follows convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_partner_name

Partner Client Age Current Input ID Follows Convention
    [Documentation]    Partner current-age input ID follows convention.
    [Tags]             naming    personal_info
    Input ID Should Follow Convention
    ...    ${MOD_PERSONAL_INFO}
    ...    input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_current

Primary Investable Assets Input ID Follows Convention
    [Documentation]    Primary investable assets input ID follows convention.
    [Tags]             naming    assets
    Input ID Should Follow Convention
    ...    ${MOD_ASSETS}
    ...    input_ID_tab_clients_subtab_clients_assets_client_primary_assets_investable

Primary Taxable Assets Input ID Follows Convention
    [Documentation]    Primary taxable assets input ID follows convention.
    [Tags]             naming    assets
    Input ID Should Follow Convention
    ...    ${MOD_ASSETS}
    ...    input_ID_tab_clients_subtab_clients_assets_client_primary_assets_taxable

Primary Tax-Deferred Assets Input ID Follows Convention
    [Documentation]    Primary tax_deferred assets input ID follows convention.
    [Tags]             naming    assets
    Input ID Should Follow Convention
    ...    ${MOD_ASSETS}
    ...    input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_deferred

Primary Tax-Free Assets Input ID Follows Convention
    [Documentation]    Primary tax_free assets input ID follows convention.
    [Tags]             naming    assets
    Input ID Should Follow Convention
    ...    ${MOD_ASSETS}
    ...    input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_free

Primary Social Security Income Input ID Follows Convention
    [Documentation]    Primary social_security income input ID follows convention.
    [Tags]             naming    income
    Input ID Should Follow Convention
    ...    ${MOD_INCOME}
    ...    input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security

Primary Pension Income Input ID Follows Convention
    [Documentation]    Primary pension income input ID follows convention.
    [Tags]             naming    income
    Input ID Should Follow Convention
    ...    ${MOD_INCOME}
    ...    input_ID_tab_clients_subtab_clients_income_client_primary_income_pension


# ---------------------------------------------------------------------------
# DEFAULT VALUES
# ---------------------------------------------------------------------------

Primary Client Name Default Is Anne Smith
    [Documentation]    Source code sets the primary client name default to "Anne Smith".
    [Tags]             defaults    personal_info
    Default Value Should Be Present In Source    ${MOD_PERSONAL_INFO}    ${DEFAULT_PRIMARY_NAME}

Partner Client Name Default Is William Smith
    [Documentation]    Source code sets the partner client name default to "William Smith".
    [Tags]             defaults    personal_info
    Default Value Should Be Present In Source    ${MOD_PERSONAL_INFO}    ${DEFAULT_PARTNER_NAME}

Primary Client Current Age Default Is 60
    [Documentation]    Source code sets the primary current-age default to 60.
    [Tags]             defaults    personal_info
    Default Value Should Be Present In Source    ${MOD_PERSONAL_INFO}    ${DEFAULT_PRIMARY_AGE_CURRENT}

Primary Client Retirement Age Default Is 65
    [Documentation]    Source code sets the primary retirement-age default to 65.
    [Tags]             defaults    personal_info
    Default Value Should Be Present In Source    ${MOD_PERSONAL_INFO}    ${DEFAULT_PRIMARY_AGE_RETIREMENT}

Marital Status Default Is Married
    [Documentation]    Source code sets the marital status default to "married".
    [Tags]             defaults    personal_info
    Default Value Should Be Present In Source    ${MOD_PERSONAL_INFO}    ${DEFAULT_MARITAL_STATUS}

Risk Tolerance Default Is Moderate
    [Documentation]    Source code sets the risk tolerance default to "moderate".
    [Tags]             defaults    personal_info
    Default Value Should Be Present In Source    ${MOD_PERSONAL_INFO}    ${DEFAULT_RISK_TOLERANCE}

Primary Client Gender Default Is Female
    [Documentation]    Source code sets the primary gender default to "female".
    [Tags]             defaults    personal_info
    Default Value Should Be Present In Source    ${MOD_PERSONAL_INFO}    ${DEFAULT_PRIMARY_GENDER}

Partner Client Gender Default Is Male
    [Documentation]    Source code sets the partner gender default to "male".
    [Tags]             defaults    personal_info
    Default Value Should Be Present In Source    ${MOD_PERSONAL_INFO}    ${DEFAULT_PARTNER_GENDER}

ZIP Code Default Is 12345
    [Documentation]    Source code sets the ZIP code default to 12345.
    [Tags]             defaults    personal_info
    Default Value Should Be Present In Source    ${MOD_PERSONAL_INFO}    ${DEFAULT_ZIP_CODE}


# ---------------------------------------------------------------------------
# CHOICE SETS
# ---------------------------------------------------------------------------

Marital Status Choices Include Single
    [Documentation]    "single" is a valid marital status choice.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    single

Marital Status Choices Include Married
    [Documentation]    "married" is a valid marital status choice.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    married

Marital Status Choices Include Divorced
    [Documentation]    "divorced" is a valid marital status choice.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    divorced

Marital Status Choices Include Widowed
    [Documentation]    "widowed" is a valid marital status choice.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    widowed

Marital Status Choices Include Domestic Partnership
    [Documentation]    "domestic_partnership" is a valid marital status choice.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    domestic_partnership

Risk Tolerance Choices Include Conservative
    [Documentation]    "conservative" is a valid risk tier.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    conservative

Risk Tolerance Choices Include Moderate Conservative
    [Documentation]    "moderate_conservative" is a valid risk tier.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    moderate_conservative

Risk Tolerance Choices Include Moderate
    [Documentation]    "moderate" is a valid risk tier.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    moderate

Risk Tolerance Choices Include Moderate Aggressive
    [Documentation]    "moderate_aggressive" is a valid risk tier.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    moderate_aggressive

Risk Tolerance Choices Include Aggressive
    [Documentation]    "aggressive" is a valid risk tier.
    [Tags]             choices    personal_info
    Module Source Should Contain    ${MOD_PERSONAL_INFO}    aggressive


# ---------------------------------------------------------------------------
# CONSTRAINT BOUNDS
# ---------------------------------------------------------------------------

Asset Constraint Upper Bound Is 100 Million
    [Documentation]    Asset values are capped at 100,000,000 per source code.
    [Tags]             constraints    assets
    Constraint Value Should Be Present    ${MOD_ASSETS}    ${ASSET_MAX}

Income Constraint Upper Bound Is 5 Million
    [Documentation]    Income values are capped at 5,000,000 per source code.
    [Tags]             constraints    income
    Constraint Value Should Be Present    ${MOD_INCOME}    ${INCOME_MAX}

Current Age Min Bound Is 18
    [Documentation]    Current age minimum is 18.
    [Tags]             constraints    personal_info
    Constraint Value Should Be Present    ${MOD_PERSONAL_INFO}    ${AGE_CURRENT_MIN}

Current Age Max Bound Is 100
    [Documentation]    Current age maximum is 100.
    [Tags]             constraints    personal_info
    Constraint Value Should Be Present    ${MOD_PERSONAL_INFO}    ${AGE_CURRENT_MAX}

Retirement Age Min Bound Is 50
    [Documentation]    Retirement age minimum is 50.
    [Tags]             constraints    personal_info
    Constraint Value Should Be Present    ${MOD_PERSONAL_INFO}    ${AGE_RETIRE_MIN}

Retirement Age Max Bound Is 80
    [Documentation]    Retirement age maximum is 80.
    [Tags]             constraints    personal_info
    Constraint Value Should Be Present    ${MOD_PERSONAL_INFO}    ${AGE_RETIRE_MAX}


# ---------------------------------------------------------------------------
# CURRENCY FORMATTING
# ---------------------------------------------------------------------------

Currency Display For Zero Is Dollar Zero
    [Documentation]    Formatting zero returns "$0".
    [Tags]             currency
    Currency Display Should Be    0    $0

Currency Display For One Thousand
    [Documentation]    Formatting 1000 returns "$1,000".
    [Tags]             currency
    Currency Display Should Be    1000    $1,000

Currency Display For One Hundred Thousand
    [Documentation]    Formatting 100000 returns "$100,000".
    [Tags]             currency
    Currency Display Should Be    100000    $100,000

Currency Display For Five Million
    [Documentation]    Formatting 5000000 returns "$5,000,000".
    [Tags]             currency
    Currency Display Should Be    5000000    $5,000,000

Currency Display Rounds Float To Whole Dollar
    [Documentation]    Formatting 1234.56 rounds to "$1,235".
    [Tags]             currency
    Currency Display Should Be    1234.56    $1,235

Currency Display Always Starts With Dollar Sign
    [Documentation]    Any valid amount starts the display with "$".
    [Tags]             currency
    Currency Display Should Start With Dollar Sign    50000


# ---------------------------------------------------------------------------
# AGE VALIDATION
# ---------------------------------------------------------------------------

Age 18 Is Valid Lower Boundary
    [Documentation]    Age 18 is at the lower boundary and should be valid.
    [Tags]             constraints    personal_info
    Age Should Be Valid    18    18    100

Age 65 Is Valid Typical Retirement Age
    [Documentation]    Age 65 is a typical retirement age and should be valid.
    [Tags]             constraints    personal_info
    Age Should Be Valid    65    18    100

Age 100 Is Valid Upper Boundary
    [Documentation]    Age 100 is at the upper boundary and should be valid.
    [Tags]             constraints    personal_info
    Age Should Be Valid    100    18    100

Age 5 Is Below Minimum And Invalid
    [Documentation]    Age 5 is below the minimum of 18 and must be rejected.
    [Tags]             constraints    personal_info
    Age Should Be Invalid    5    18    100

Age 150 Is Above Maximum And Invalid
    [Documentation]    Age 150 is above the maximum of 100 and must be rejected.
    [Tags]             constraints    personal_info
    Age Should Be Invalid    150    18    100


# ---------------------------------------------------------------------------
# FINANCIAL AMOUNT VALIDATION
# ---------------------------------------------------------------------------

Zero Is A Valid Financial Amount
    [Documentation]    Zero is an acceptable starting value for any financial field.
    [Tags]             constraints    assets    income
    Financial Amount Should Be Valid    0

Positive Amount Is Valid
    [Documentation]    A typical portfolio value passes financial validation.
    [Tags]             constraints    assets
    Financial Amount Should Be Valid    250000

Negative Amount Is Invalid
    [Documentation]    Negative portfolio/income values must be rejected.
    [Tags]             constraints    assets    income
    Financial Amount Should Be Invalid    -1


# ---------------------------------------------------------------------------
# TAB CLIENTS ORCHESTRATION
# ---------------------------------------------------------------------------

Tab Clients Server References Personal Info Return Key
    [Documentation]    tab_clients_server returns key for personal_info subtab.
    [Tags]             integration    smoke
    Module Source Should Contain    ${MOD_TAB_CLIENTS}    clients_Personal_Info_Server

Tab Clients Server References Assets Return Key
    [Documentation]    tab_clients_server returns key for assets subtab.
    [Tags]             integration    smoke
    Module Source Should Contain    ${MOD_TAB_CLIENTS}    clients_Assets_Server

Tab Clients Server References Goals Return Key
    [Documentation]    tab_clients_server returns key for goals subtab.
    [Tags]             integration    smoke
    Module Source Should Contain    ${MOD_TAB_CLIENTS}    clients_Goals_Server

Tab Clients Server References Income Return Key
    [Documentation]    tab_clients_server returns key for income subtab.
    [Tags]             integration    smoke
    Module Source Should Contain    ${MOD_TAB_CLIENTS}    clients_Income_Server

Tab Clients Server References Summary Return Key
    [Documentation]    tab_clients_server returns key for summary subtab.
    [Tags]             integration    smoke
    Module Source Should Contain    ${MOD_TAB_CLIENTS}    clients_Summary_Server

Tab Clients UI Defines Top Level NavSet ID
    [Documentation]    tab_clients_ui sets navset ID to "ID_tab_clients_tabs_all".
    [Tags]             integration    smoke    naming
    Module Source Should Contain    ${MOD_TAB_CLIENTS}    ID_tab_clients_tabs_all
