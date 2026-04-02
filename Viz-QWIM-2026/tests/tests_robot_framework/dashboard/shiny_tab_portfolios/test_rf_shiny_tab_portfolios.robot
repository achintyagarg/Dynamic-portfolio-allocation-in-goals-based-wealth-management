*** Settings ***
Documentation     Robot Framework tests for the ``shiny_tab_portfolios`` dashboard module.
...
...               These tests validate:
...               - Input ID naming conventions across all four subtabs
...               - Module API surface (callable UI and server functions)
...               - skfolio optimization method constants
...               - Optimization categories and method counts
...               - OUTPUT_DIR configuration in all modules
...               - Server function parameter signatures
...               - Default values encoded in source
...
...               No live Shiny server is required; all tests operate on
...               pure-Python module logic via the keyword library.
...
...               Run:
...                   robot --outputdir results/robot_all tests/tests_robot_framework/dashboard/shiny_tab_portfolios/
...
...               Tags used:
...               ``portfolios_tab``   -- All tests in this suite
...               ``smoke``            -- Fast module-import / API surface tests
...               ``naming``           -- Input ID convention tests
...               ``constants``        -- Optimization constant tests
...               ``defaults``         -- Default value tests
...               ``integration``      -- Cross-module interaction tests
...
...               Author:  QWIM Development Team
...               Version: 0.2.0

Library           ${CURDIR}${/}keywords_shiny_tab_portfolios.py

Suite Setup       Log    Starting shiny_tab_portfolios Robot Framework test suite
Suite Teardown    Log    Finished shiny_tab_portfolios Robot Framework test suite

Force Tags        portfolios_tab


*** Variables ***
# Module short names
${MOD_ANALYSIS}       subtab_portfolios_analysis
${MOD_COMPARISON}     subtab_portfolios_comparison
${MOD_WEIGHTS}        subtab_weights_analysis
${MOD_SKFOLIO}        subtab_portfolios_skfolio
${MOD_TAB}            tab_portfolios

# Expected constant counts
${COUNT_CATEGORIES}         4
${COUNT_BASIC_METHODS}      3
${COUNT_CONVEX_METHODS}     5
${COUNT_CLUSTERING_METHODS} 4
${COUNT_ENSEMBLE_METHODS}   1
${COUNT_OBJECTIVE_FUNCS}    4
${COUNT_TOTAL_METHODS}      13

# Input ID prefixes
${PREFIX_ANALYSIS}       input_ID_tab_portfolios_subtab_portfolios_analysis_
${PREFIX_COMPARISON}     input_ID_tab_portfolios_subtab_portfolios_comparison_
${PREFIX_WEIGHTS}        input_ID_tab_portfolios_subtab_weights_analysis_
${PREFIX_SKFOLIO}        input_ID_tab_portfolios_subtab_skfolio_

# Method key prefixes
${BASIC_KEY_PREFIX}       BASIC_
${CONVEX_KEY_PREFIX}      CONVEX_
${CLUSTER_KEY_PREFIX}     CLUSTERING_
${ENSEMBLE_KEY_PREFIX}    ENSEMBLE_

# Default values
${DEFAULT_ANALYSIS_TIME_PERIOD}    1y
${DEFAULT_ANALYSIS_TYPE}           returns
${DEFAULT_SKFOLIO_TIME_PERIOD}     3y
${DEFAULT_SKFOLIO_METHOD1}         basic
${DEFAULT_SKFOLIO_METHOD2}         convex


*** Test Cases ***

# ---------------------------------------------------------------------------
# SMOKE — Module API surface
# ---------------------------------------------------------------------------

All Subtab UI Functions Are Callable
    [Documentation]    Every subtab module exposes a callable UI function.
    [Tags]             smoke    api_surface
    Module Should Expose Callable    ${MOD_ANALYSIS}     subtab_portfolios_analysis_ui
    Module Should Expose Callable    ${MOD_COMPARISON}   subtab_portfolios_comparison_ui
    Module Should Expose Callable    ${MOD_WEIGHTS}      subtab_weights_analysis_ui
    Module Should Expose Callable    ${MOD_SKFOLIO}      subtab_portfolios_skfolio_ui
    Module Should Expose Callable    ${MOD_TAB}          tab_portfolios_ui

All Subtab Server Functions Are Callable
    [Documentation]    Every subtab module exposes a callable server function.
    [Tags]             smoke    api_surface
    Module Should Expose Callable    ${MOD_ANALYSIS}     subtab_portfolios_analysis_server
    Module Should Expose Callable    ${MOD_COMPARISON}   subtab_portfolios_comparison_server
    Module Should Expose Callable    ${MOD_WEIGHTS}      subtab_weights_analysis_server
    Module Should Expose Callable    ${MOD_SKFOLIO}      subtab_portfolios_skfolio_server
    Module Should Expose Callable    ${MOD_TAB}          tab_portfolios_server

All Modules Have OUTPUT_DIR
    [Documentation]    Analysis, skfolio and tab modules define OUTPUT_DIR.
    [Tags]             smoke    constants
    Module OUTPUT_DIR Should Be Path    ${MOD_ANALYSIS}
    Module OUTPUT_DIR Should Be Path    ${MOD_SKFOLIO}
    Module OUTPUT_DIR Should Be Path    ${MOD_TAB}

skfolio Module Has All Method Dicts
    [Documentation]    skfolio module exposes all five constant dicts.
    [Tags]             smoke    constants
    Module Should Have Attribute    ${MOD_SKFOLIO}    OPTIMIZATION_CATEGORIES
    Module Should Have Attribute    ${MOD_SKFOLIO}    BASIC_METHODS
    Module Should Have Attribute    ${MOD_SKFOLIO}    CONVEX_METHODS
    Module Should Have Attribute    ${MOD_SKFOLIO}    CLUSTERING_METHODS
    Module Should Have Attribute    ${MOD_SKFOLIO}    ENSEMBLE_METHODS
    Module Should Have Attribute    ${MOD_SKFOLIO}    OBJECTIVE_FUNCTIONS


# ---------------------------------------------------------------------------
# NAMING — Input ID conventions
# ---------------------------------------------------------------------------

Analysis Input IDs Follow Hierarchical Convention
    [Documentation]    All analysis subtab input IDs start with the required prefix.
    [Tags]             naming
    Input IDs Should Follow Naming Convention    ${MOD_ANALYSIS}    ${PREFIX_ANALYSIS}

Comparison Input IDs Follow Hierarchical Convention
    [Documentation]    All comparison subtab input IDs start with the required prefix.
    [Tags]             naming
    Input IDs Should Follow Naming Convention    ${MOD_COMPARISON}    ${PREFIX_COMPARISON}

Weights Analysis Input IDs Follow Hierarchical Convention
    [Documentation]    All weights subtab input IDs start with the required prefix.
    [Tags]             naming
    Input IDs Should Follow Naming Convention    ${MOD_WEIGHTS}    ${PREFIX_WEIGHTS}

skfolio Input IDs Follow Hierarchical Convention
    [Documentation]    All skfolio subtab input IDs start with the required prefix.
    [Tags]             naming
    Input IDs Should Follow Naming Convention    ${MOD_SKFOLIO}    ${PREFIX_SKFOLIO}


# ---------------------------------------------------------------------------
# CONSTANTS — skfolio optimization method counts
# ---------------------------------------------------------------------------

OPTIMIZATION_CATEGORIES Has Exactly Four Entries
    [Documentation]    OPTIMIZATION_CATEGORIES dict has 4 entries.
    [Tags]             constants
    Dict Should Have Length    ${MOD_SKFOLIO}    OPTIMIZATION_CATEGORIES    ${COUNT_CATEGORIES}

BASIC_METHODS Has Exactly Three Entries
    [Documentation]    BASIC_METHODS dict has 3 entries.
    [Tags]             constants
    Dict Should Have Length    ${MOD_SKFOLIO}    BASIC_METHODS    ${COUNT_BASIC_METHODS}

CONVEX_METHODS Has Exactly Five Entries
    [Documentation]    CONVEX_METHODS dict has 5 entries.
    [Tags]             constants
    Dict Should Have Length    ${MOD_SKFOLIO}    CONVEX_METHODS    ${COUNT_CONVEX_METHODS}

CLUSTERING_METHODS Has Exactly Four Entries
    [Documentation]    CLUSTERING_METHODS dict has 4 entries.
    [Tags]             constants
    Dict Should Have Length    ${MOD_SKFOLIO}    CLUSTERING_METHODS    ${COUNT_CLUSTERING_METHODS}

ENSEMBLE_METHODS Has Exactly One Entry
    [Documentation]    ENSEMBLE_METHODS dict has 1 entry.
    [Tags]             constants
    Dict Should Have Length    ${MOD_SKFOLIO}    ENSEMBLE_METHODS    ${COUNT_ENSEMBLE_METHODS}

OBJECTIVE_FUNCTIONS Has Exactly Four Entries
    [Documentation]    OBJECTIVE_FUNCTIONS dict has 4 entries.
    [Tags]             constants
    Dict Should Have Length    ${MOD_SKFOLIO}    OBJECTIVE_FUNCTIONS    ${COUNT_OBJECTIVE_FUNCS}

Total Method Count Is Thirteen
    [Documentation]    BASIC + CONVEX + CLUSTERING + ENSEMBLE = 13 total methods.
    [Tags]             constants
    Total Method Count Should Be    ${MOD_SKFOLIO}    ${COUNT_TOTAL_METHODS}

No Duplicate Method Keys Across Categories
    [Documentation]    No method key appears in more than one category dict.
    [Tags]             constants
    No Duplicate Keys Across Method Dicts    ${MOD_SKFOLIO}

Each Optimization Category Has Non Empty Method Dict
    [Documentation]    Every key in OPTIMIZATION_CATEGORIES maps to a non-empty method dict.
    [Tags]             constants    integration
    Each Category Should Have Non Empty Method Dict    ${MOD_SKFOLIO}

BASIC_METHODS Keys Have Correct Prefix
    [Documentation]    All BASIC_METHODS keys start with 'BASIC_'.
    [Tags]             constants    naming
    All Keys In Dict Should Start With Prefix    ${MOD_SKFOLIO}    BASIC_METHODS    ${BASIC_KEY_PREFIX}

CONVEX_METHODS Keys Have Correct Prefix
    [Documentation]    All CONVEX_METHODS keys start with 'CONVEX_'.
    [Tags]             constants    naming
    All Keys In Dict Should Start With Prefix    ${MOD_SKFOLIO}    CONVEX_METHODS    ${CONVEX_KEY_PREFIX}

CLUSTERING_METHODS Keys Have Correct Prefix
    [Documentation]    All CLUSTERING_METHODS keys start with 'CLUSTERING_'.
    [Tags]             constants    naming
    All Keys In Dict Should Start With Prefix    ${MOD_SKFOLIO}    CLUSTERING_METHODS    ${CLUSTER_KEY_PREFIX}

ENSEMBLE_METHODS Keys Have Correct Prefix
    [Documentation]    All ENSEMBLE_METHODS keys start with 'ENSEMBLE_'.
    [Tags]             constants    naming
    All Keys In Dict Should Start With Prefix    ${MOD_SKFOLIO}    ENSEMBLE_METHODS    ${ENSEMBLE_KEY_PREFIX}


# ---------------------------------------------------------------------------
# Signatures — server function parameters
# ---------------------------------------------------------------------------

Analysis Server Accepts id Parameter
    [Documentation]    subtab_portfolios_analysis_server accepts 'id' parameter.
    [Tags]             smoke    api_surface
    Server Function Should Accept Parameter
    ...    ${MOD_ANALYSIS}    subtab_portfolios_analysis_server    id

Comparison Server Accepts id Parameter
    [Documentation]    subtab_portfolios_comparison_server accepts 'id' parameter.
    [Tags]             smoke    api_surface
    Server Function Should Accept Parameter
    ...    ${MOD_COMPARISON}    subtab_portfolios_comparison_server    id

Weights Server Accepts id Parameter
    [Documentation]    subtab_weights_analysis_server accepts 'id' parameter.
    [Tags]             smoke    api_surface
    Server Function Should Accept Parameter
    ...    ${MOD_WEIGHTS}    subtab_weights_analysis_server    id

skfolio Server Accepts id Parameter
    [Documentation]    subtab_portfolios_skfolio_server accepts 'id' parameter.
    [Tags]             smoke    api_surface
    Server Function Should Accept Parameter
    ...    ${MOD_SKFOLIO}    subtab_portfolios_skfolio_server    id

Tab Portfolios Server Accepts id Parameter
    [Documentation]    tab_portfolios_server accepts 'id' parameter.
    [Tags]             smoke    api_surface
    Server Function Should Accept Parameter
    ...    ${MOD_TAB}    tab_portfolios_server    id


# ---------------------------------------------------------------------------
# DEFAULTS — encoded default values
# ---------------------------------------------------------------------------

Analysis Time Period Default Is 1y
    [Documentation]    Portfolio analysis time period defaults to '1y'.
    [Tags]             defaults
    Default Value In Source Should Be
    ...    ${MOD_ANALYSIS}
    ...    input_ID_tab_portfolios_subtab_portfolios_analysis_time_period
    ...    ${DEFAULT_ANALYSIS_TIME_PERIOD}

skfolio Time Period Default Is 3y
    [Documentation]    skfolio time period defaults to '3y'.
    [Tags]             defaults
    Default Value In Source Should Be
    ...    ${MOD_SKFOLIO}
    ...    input_ID_tab_portfolios_subtab_skfolio_time_period
    ...    ${DEFAULT_SKFOLIO_TIME_PERIOD}

skfolio Method1 Category Default Is basic
    [Documentation]    skfolio method1 category defaults to 'basic'.
    [Tags]             defaults
    Default Value In Source Should Be
    ...    ${MOD_SKFOLIO}
    ...    input_ID_tab_portfolios_subtab_skfolio_method1_category
    ...    ${DEFAULT_SKFOLIO_METHOD1}

skfolio Method2 Category Default Is convex
    [Documentation]    skfolio method2 category defaults to 'convex'.
    [Tags]             defaults
    Default Value In Source Should Be
    ...    ${MOD_SKFOLIO}
    ...    input_ID_tab_portfolios_subtab_skfolio_method2_category
    ...    ${DEFAULT_SKFOLIO_METHOD2}
