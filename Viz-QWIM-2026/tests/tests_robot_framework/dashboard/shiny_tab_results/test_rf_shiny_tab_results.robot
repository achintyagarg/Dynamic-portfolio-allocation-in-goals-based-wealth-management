*** Settings ***
Documentation     Robot Framework tests for the ``shiny_tab_results`` dashboard module.
...
...               These tests validate:
...               - Module API surface (callable UI and server functions)
...               - Input ID naming conventions
...               - Security constants in subtab_reporting
...               - sanitize_filename_for_security helper behaviour
...               - Simulation constants in subtab_simulation
...
...               No live Shiny server is required; all tests operate on
...               pure-Python module logic via the keyword library.
...
...               Run:
...                   robot --outputdir results/robot_all tests/tests_robot_framework/dashboard/shiny_tab_results/
...
...               Tags used:
...               ``results_tab``   -- All tests in this suite
...               ``smoke``         -- Fast module-import / API surface tests
...               ``naming``        -- Input ID convention tests
...               ``security``      -- Security constant and sanitization tests
...               ``simulation``    -- Simulation constant tests
...
...               Author:  QWIM Development Team
...               Version: 0.1.0

Library           ${CURDIR}${/}keywords_shiny_tab_results.py

Suite Setup       Log    Starting shiny_tab_results Robot Framework test suite
Suite Teardown    Log    Finished shiny_tab_results Robot Framework test suite

Force Tags        results_tab


*** Variables ***
# Module short names
${MOD_TAB}            tab_results
${MOD_REPORTING}      subtab_reporting
${MOD_SIMULATION}     subtab_simulation

# Naming convention prefixes
${PREFIX_REPORTING}     input_ID_tab_results_subtab_reporting
${PREFIX_SIMULATION}    input_ID_tab_results_subtab_simulation

# Security constant names
${CONST_MAX_LEN}      MAX_FILENAME_LENGTH
${CONST_FORBIDDEN}    FORBIDDEN_FILENAME_PARTS
${CONST_PATTERN}      ALLOWED_FILENAME_PATTERN

# Simulation constant names
${CONST_ALL_ETFS}      ALL_ETF_SYMBOLS
${CONST_DEF_ETFS}      DEFAULT_SELECTED_ETFS
${CONST_DIST}          DISTRIBUTION_CHOICES
${CONST_RNG}           RNG_TYPE_CHOICES


*** Test Cases ***

# ---------------------------------------------------------------------------
# Smoke — module API surface
# ---------------------------------------------------------------------------

tab_results UI Function Is Callable
    [Tags]    smoke    api
    Module Should Expose Callable    ${MOD_TAB}    tab_results_ui

tab_results Server Function Is Callable
    [Tags]    smoke    api
    Module Should Expose Callable    ${MOD_TAB}    tab_results_server

subtab_reporting UI Function Is Callable
    [Tags]    smoke    api
    Module Should Expose Callable    ${MOD_REPORTING}    subtab_reporting_ui

subtab_reporting Server Function Is Callable
    [Tags]    smoke    api
    Module Should Expose Callable    ${MOD_REPORTING}    subtab_reporting_server

subtab_simulation UI Function Is Callable
    [Tags]    smoke    api
    Module Should Expose Callable    ${MOD_SIMULATION}    subtab_simulation_ui

subtab_simulation Server Function Is Callable
    [Tags]    smoke    api
    Module Should Expose Callable    ${MOD_SIMULATION}    subtab_simulation_server

# ---------------------------------------------------------------------------
# Naming — input ID convention
# ---------------------------------------------------------------------------

subtab_reporting Input IDs Follow Convention
    [Tags]    naming
    Input IDs Should Follow Naming Convention    ${MOD_REPORTING}    ${PREFIX_REPORTING}

subtab_simulation Input IDs Follow Convention
    [Tags]    naming
    Input IDs Should Follow Naming Convention    ${MOD_SIMULATION}    ${PREFIX_SIMULATION}

# ---------------------------------------------------------------------------
# Security constants — subtab_reporting
# ---------------------------------------------------------------------------

MAX_FILENAME_LENGTH Is A Positive Integer
    [Tags]    smoke    security
    Security Constant Should Be Positive Integer    ${MOD_REPORTING}    ${CONST_MAX_LEN}

FORBIDDEN_FILENAME_PARTS Contains Path Traversal
    [Tags]    security
    List Constant Should Contain Value    ${MOD_REPORTING}    ${CONST_FORBIDDEN}    ..

FORBIDDEN_FILENAME_PARTS Contains Backslash
    [Tags]    security
    List Constant Should Contain Value    ${MOD_REPORTING}    ${CONST_FORBIDDEN}    \\

FORBIDDEN_FILENAME_PARTS Contains Dollar Sign
    [Tags]    security
    List Constant Should Contain Value    ${MOD_REPORTING}    ${CONST_FORBIDDEN}    $

FORBIDDEN_FILENAME_PARTS Contains Percent Sign
    [Tags]    security
    List Constant Should Contain Value    ${MOD_REPORTING}    ${CONST_FORBIDDEN}    %

ALLOWED_FILENAME_PATTERN Is A Compiled Regex
    [Tags]    smoke    security
    Attribute Should Be Compiled Regex    ${MOD_REPORTING}    ${CONST_PATTERN}

# ---------------------------------------------------------------------------
# sanitize_filename_for_security — acceptance
# ---------------------------------------------------------------------------

Well-Formed PDF Filename Returns 3-Tuple
    [Tags]    smoke    security
    Sanitize Filename Should Return Tuple    ${MOD_REPORTING}    QWIM_Report_2024.pdf

Well-Formed PDF Filename Is Accepted
    [Tags]    smoke    security
    Filename Should Be Accepted    ${MOD_REPORTING}    QWIM_Report_2024.pdf

Filename With Spaces Is Accepted
    [Tags]    security
    Filename Should Be Accepted    ${MOD_REPORTING}    Annual Report 2024.pdf

Filename With Parentheses Is Accepted
    [Tags]    security
    Filename Should Be Accepted    ${MOD_REPORTING}    Portfolio (Final) 2024.pdf

# ---------------------------------------------------------------------------
# sanitize_filename_for_security — rejection
# ---------------------------------------------------------------------------

Empty Filename Is Rejected
    [Tags]    smoke    security
    Filename Should Be Rejected    ${MOD_REPORTING}    ${EMPTY}

Filename Without PDF Extension Is Rejected
    [Tags]    security
    Filename Should Be Rejected    ${MOD_REPORTING}    Report_2024.docx

Filename With Path Traversal Is Rejected
    [Tags]    smoke    security
    Filename Should Be Rejected    ${MOD_REPORTING}    ../../etc/passwd.pdf

Filename Starting With Dot Is Rejected
    [Tags]    security
    Filename Should Be Rejected    ${MOD_REPORTING}    .hidden_file.pdf

Filename With Dollar Sign Is Rejected
    [Tags]    security
    Filename Should Be Rejected    ${MOD_REPORTING}    Report$2024.pdf

Filename With Percent Sign Is Rejected
    [Tags]    security
    Filename Should Be Rejected    ${MOD_REPORTING}    Report%2024.pdf

Filename With Tilde Is Rejected
    [Tags]    security
    Filename Should Be Rejected    ${MOD_REPORTING}    Report~2024.pdf

Extremely Long Filename Is Rejected
    [Tags]    security
    # Build a 300-char filename (well above MAX_FILENAME_LENGTH=200)
    ${long_filename}=    Evaluate    "x" * 295 + ".pdf"
    Filename Should Be Rejected    ${MOD_REPORTING}    ${long_filename}

# ---------------------------------------------------------------------------
# Simulation constants — subtab_simulation
# ---------------------------------------------------------------------------

ALL_ETF_SYMBOLS Has 12 Entries
    [Tags]    smoke    simulation
    Sequence Constant Should Have Length    ${MOD_SIMULATION}    ${CONST_ALL_ETFS}    12

DEFAULT_SELECTED_ETFS Has 3 Entries
    [Tags]    simulation
    Sequence Constant Should Have Length    ${MOD_SIMULATION}    ${CONST_DEF_ETFS}    3

Default ETFs Are All In ALL_ETF_SYMBOLS
    [Tags]    simulation
    All Items In Constant Should Be In Other Constant
    ...    ${MOD_SIMULATION}    ${CONST_DEF_ETFS}    ${CONST_ALL_ETFS}

DISTRIBUTION_CHOICES Has 3 Keys
    [Tags]    simulation
    Dict Constant Should Have Length    ${MOD_SIMULATION}    ${CONST_DIST}    3

DISTRIBUTION_CHOICES Contains normal
    [Tags]    smoke    simulation
    Dict Constant Should Contain Key    ${MOD_SIMULATION}    ${CONST_DIST}    normal

DISTRIBUTION_CHOICES Contains lognormal
    [Tags]    simulation
    Dict Constant Should Contain Key    ${MOD_SIMULATION}    ${CONST_DIST}    lognormal

DISTRIBUTION_CHOICES Contains student_t
    [Tags]    simulation
    Dict Constant Should Contain Key    ${MOD_SIMULATION}    ${CONST_DIST}    student_t

RNG_TYPE_CHOICES Has 4 Keys
    [Tags]    simulation
    Dict Constant Should Have Length    ${MOD_SIMULATION}    ${CONST_RNG}    4

RNG_TYPE_CHOICES Contains pcg64
    [Tags]    smoke    simulation
    Dict Constant Should Contain Key    ${MOD_SIMULATION}    ${CONST_RNG}    pcg64

RNG_TYPE_CHOICES Contains mt19937
    [Tags]    simulation
    Dict Constant Should Contain Key    ${MOD_SIMULATION}    ${CONST_RNG}    mt19937

RNG_TYPE_CHOICES Contains philox
    [Tags]    simulation
    Dict Constant Should Contain Key    ${MOD_SIMULATION}    ${CONST_RNG}    philox

RNG_TYPE_CHOICES Contains sfc64
    [Tags]    simulation
    Dict Constant Should Contain Key    ${MOD_SIMULATION}    ${CONST_RNG}    sfc64

# ---------------------------------------------------------------------------
# Security function module-level availability
# ---------------------------------------------------------------------------

subtab_reporting Exposes sanitize_filename_for_security
    [Tags]    smoke    security
    Module Should Expose Callable    ${MOD_REPORTING}    sanitize_filename_for_security

subtab_reporting Exposes create_secure_temp_directory
    [Tags]    smoke    security
    Module Should Expose Callable    ${MOD_REPORTING}    create_secure_temp_directory

subtab_reporting Exposes cleanup_temp_files
    [Tags]    smoke    security
    Module Should Expose Callable    ${MOD_REPORTING}    cleanup_temp_files
