*** Settings ***
Documentation
...    Robot Framework test suite for utils_tab_results (shiny_utils).
...
...    Verifies: process_data_for_plot_tab_results passes small DataFrames
...    through unchanged and downsamples large ones; normalize_data_tab_results
...    leaves data unchanged for 'none', scales to [0,1] for 'min_max', and
...    returns a DataFrame for 'z_score'; transform_data_tab_results leaves
...    data unchanged for 'none', adds _pct columns for 'percent_change', and
...    adds monotonically increasing _cum columns for 'cumulative'.
...
...    Test Categories:
...    - process_data_for_plot_tab_results (passthrough / downsample)
...    - normalize_data_tab_results (none / min_max / z_score)
...    - transform_data_tab_results (none / percent_change / cumulative)
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-02-28

Library     ${CURDIR}${/}keywords_utils_data.py
Library     Collections

Suite Setup      Log    Starting Utils Tab Results Robot Framework Test Suite
Suite Teardown   Log    Utils Tab Results Test Suite Complete


*** Variables ***
${SMALL_DF_ROWS}        ${20}
${NORMALISED_ROWS}      ${20}


*** Test Cases ***

# ---------------------------------------------------------------------------
# process_data_for_plot_tab_results
# ---------------------------------------------------------------------------

Small DataFrame Passes Through Process Without Change
    [Documentation]    A DataFrame with 20 rows (well below max_points=5000)
    ...    is returned unchanged by process_data_for_plot_tab_results.
    [Tags]    tab_results    process    smoke
    ${result}=    Process Small Dataframe For Plot
    Processed Dataframe Should Equal Input For Small Data    ${result}

Large DataFrame Is Downsampled To Fewer Rows
    [Documentation]    A 200-row DataFrame processed with max_points=50 is
    ...    returned with fewer than 200 rows.
    [Tags]    tab_results    process
    Process Large Dataframe Returns Fewer Rows


# ---------------------------------------------------------------------------
# normalize_data_tab_results — none
# ---------------------------------------------------------------------------

None Method Returns Unchanged DataFrame
    [Documentation]    Normalising with method='none' returns the DataFrame
    ...    without modification.
    [Tags]    tab_results    normalise    smoke
    ${result}=    Normalize With None Method Returns Unchanged
    Normalised Dataframe Should Be Valid    ${result}    ${NORMALISED_ROWS}


# ---------------------------------------------------------------------------
# normalize_data_tab_results — min_max
# ---------------------------------------------------------------------------

Min Max Normalisation Scales Values To Zero One Range
    [Documentation]    After min_max normalisation all numeric values are
    ...    within [0.0, 1.0].
    [Tags]    tab_results    normalise
    Normalize With Min Max Returns Values In Zero One Range    ${NORMALISED_ROWS}


# ---------------------------------------------------------------------------
# normalize_data_tab_results — z_score
# ---------------------------------------------------------------------------

Z Score Normalisation Returns Valid DataFrame
    [Documentation]    z_score normalisation returns a non-empty DataFrame with
    ...    the correct row count.
    [Tags]    tab_results    normalise
    ${result}=    Normalize With Z Score Returns Dataframe
    Normalised Dataframe Should Be Valid    ${result}    ${NORMALISED_ROWS}


# ---------------------------------------------------------------------------
# transform_data_tab_results — none
# ---------------------------------------------------------------------------

None Transformation Returns Unchanged DataFrame
    [Documentation]    Transforming with transformation='none' returns the
    ...    DataFrame without modification.
    [Tags]    tab_results    transform    smoke
    ${result}=    Transform With None Returns Dataframe Unchanged
    Should Not Be Equal    ${result}    ${None}

# ---------------------------------------------------------------------------
# transform_data_tab_results — percent_change
# ---------------------------------------------------------------------------

Percent Change Transformation Adds Pct Suffix Columns
    [Documentation]    After percent_change transformation the DataFrame
    ...    contains at least one column with a '_pct' suffix.
    [Tags]    tab_results    transform
    ${result}=    Transform With Percent Change Adds Pct Columns
    Transformed Dataframe Should Contain Pct Columns    ${result}

Percent Change DataFrame Is Non-Empty After Transform
    [Documentation]    The DataFrame returned by percent_change transformation
    ...    is not empty.
    [Tags]    tab_results    transform
    ${result}=    Transform With Percent Change Adds Pct Columns
    Should Not Be Equal    ${result}    ${None}
    ${length}=    Evaluate    len($result)
    Should Be True    ${length} > 0


# ---------------------------------------------------------------------------
# transform_data_tab_results — cumulative
# ---------------------------------------------------------------------------

Cumulative Transformation Adds Cum Suffix Columns
    [Documentation]    After cumulative transformation the DataFrame contains
    ...    at least one column with a '_cum' suffix.
    [Tags]    tab_results    transform
    ${result}=    Transform With Cumulative Adds Cum Columns
    Transformed Dataframe Should Contain Cum Columns    ${result}

Cumulative Columns Are Monotonically Non-Decreasing
    [Documentation]    Cumulative sum of positive-valued series must be
    ...    non-decreasing throughout.
    [Tags]    tab_results    transform
    ${result}=    Transform With Cumulative Adds Cum Columns
    Cumulative Values Should Be Monotonically Increasing    ${result}
