*** Settings ***
Documentation
...    Robot Framework test suite for the reporting package public API.
...
...    Verifies the ``src.dashboard.reporting`` export surface after the
...    2026-01 pyright cleanup that removed ``build_typst_data_context``
...    from ``reporting/__init__.py``.
...
...    Test Categories:
...    - Package importability smoke
...    - Positive export checks (3 public symbols)
...    - Negative regression guard (removed symbol absent)
...    - __all__ completeness
...    - validate_polars_DF functional smoke
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:  2026-01

Library     ${CURDIR}${/}keywords_reporting_exports.py
Library     Collections

Suite Setup      Log    Starting Reporting Package API Test Suite
Suite Teardown   Log    Reporting Package API Test Suite Complete


*** Test Cases ***

# ---------------------------------------------------------------------------
# Package importability
# ---------------------------------------------------------------------------

Reporting Package Is Importable
    [Documentation]    src.dashboard.reporting must be importable in this environment.
    [Tags]    reporting    api    smoke
    Reporting Package Should Be Importable

# ---------------------------------------------------------------------------
# Positive export checks
# ---------------------------------------------------------------------------

Compile Typst Report Is Exported
    [Documentation]    compile_typst_report must be accessible from the reporting package.
    [Tags]    reporting    api    smoke
    Symbol Should Be Exported    compile_typst_report

Compile Typst Report Is Callable
    [Documentation]    compile_typst_report must be a callable.
    [Tags]    reporting    api    smoke
    Symbol Should Be Callable    compile_typst_report

Generate Report PDF Is Exported
    [Documentation]    generate_report_PDF must be accessible from the reporting package.
    [Tags]    reporting    api    smoke
    Symbol Should Be Exported    generate_report_PDF

Generate Report PDF Is Callable
    [Documentation]    generate_report_PDF must be a callable.
    [Tags]    reporting    api    smoke
    Symbol Should Be Callable    generate_report_PDF

Validate Polars DF Is Exported
    [Documentation]    validate_polars_DF must be accessible from the reporting package.
    [Tags]    reporting    api    smoke
    Symbol Should Be Exported    validate_polars_DF

Validate Polars DF Is Callable
    [Documentation]    validate_polars_DF must be a callable.
    [Tags]    reporting    api    smoke
    Symbol Should Be Callable    validate_polars_DF

# ---------------------------------------------------------------------------
# Negative regression guard
# ---------------------------------------------------------------------------

Build Typst Data Context Is NOT Exported
    [Documentation]
    ...    build_typst_data_context must NOT be accessible from src.dashboard.reporting.
    ...
    ...    Regression guard for the 2026-01 fix: this function never existed in
    ...    report_QWIM.py and its inclusion in __init__.py was a bug that caused
    ...    ImportError at consumer sites and silently skipped 56 unit tests.
    [Tags]    reporting    api    regression
    Symbol Should NOT Be Exported    build_typst_data_context

Build Typst Data Context Not In All List
    [Documentation]    build_typst_data_context must not appear in reporting.__all__.
    [Tags]    reporting    api    regression
    All List Should Not Contain    build_typst_data_context

# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------

All List Has Exactly Three Entries
    [Documentation]    reporting.__all__ must contain exactly 3 public symbols.
    [Tags]    reporting    api    regression
    All List Should Contain Exactly    3

All List Contains Compile Typst Report
    [Documentation]    compile_typst_report must appear in reporting.__all__.
    [Tags]    reporting    api    regression
    All List Should Contain    compile_typst_report

All List Contains Generate Report PDF
    [Documentation]    generate_report_PDF must appear in reporting.__all__.
    [Tags]    reporting    api    regression
    All List Should Contain    generate_report_PDF

All List Contains Validate Polars DF
    [Documentation]    validate_polars_DF must appear in reporting.__all__.
    [Tags]    reporting    api    regression
    All List Should Contain    validate_polars_DF

# ---------------------------------------------------------------------------
# validate_polars_DF functional smoke
# ---------------------------------------------------------------------------

Validate Polars DF Accepts Valid DataFrame
    [Documentation]    validate_polars_DF should not return False for a well-formed DataFrame.
    [Tags]    reporting    functional    smoke
    Validate Polars Df With Valid Dataframe

Validate Polars DF Handles None Without Crash
    [Documentation]    validate_polars_DF must not raise an unhandled exception for None.
    [Tags]    reporting    functional    edge_case
    Validate Polars Df With None Input
