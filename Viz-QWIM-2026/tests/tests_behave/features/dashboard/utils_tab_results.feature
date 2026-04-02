Feature: Dashboard utils_tab_results — data processing, normalisation and transformation
  As a dashboard developer
  I want reliable tab-results data utilities
  So that charts display well-prepared data

  # -------------------------------------------------------------------------
  # process_data_for_plot_tab_results
  # -------------------------------------------------------------------------

  @utils_tab_results @process @smoke
  Scenario: Small DataFrame passes through plot processing unchanged
    When I process a small 20-row DataFrame for plotting
    Then the processed DataFrame should have 20 rows

  @utils_tab_results @process
  Scenario: Large DataFrame is downsampled to fewer rows
    When I process a large 200-row DataFrame with max_points 50
    Then the processed DataFrame should have fewer than 200 rows

  # -------------------------------------------------------------------------
  # normalize_data_tab_results
  # -------------------------------------------------------------------------

  @utils_tab_results @normalise @smoke
  Scenario: None normalisation method returns data unchanged
    When I normalise a 10-row DataFrame with method "none"
    Then the normalised DataFrame should have 10 rows

  @utils_tab_results @normalise
  Scenario: Min-max normalisation scales values to the zero-one range
    When I normalise a 20-row DataFrame with method "min_max"
    Then all numeric column values should be in the range 0.0 to 1.0

  @utils_tab_results @normalise
  Scenario: Z-score normalisation returns a valid DataFrame
    When I normalise a 20-row DataFrame with method "z_score"
    Then the normalised DataFrame should have 20 rows

  # -------------------------------------------------------------------------
  # transform_data_tab_results
  # -------------------------------------------------------------------------

  @utils_tab_results @transform @smoke
  Scenario: None transformation returns data unchanged
    When I transform a 10-row DataFrame with transformation "none"
    Then the transformed DataFrame should have 10 rows

  @utils_tab_results @transform
  Scenario: Percent-change transformation adds _pct suffix columns
    When I transform a 10-row DataFrame with transformation "percent_change"
    Then the transformed DataFrame should contain columns with "_pct" suffix

  @utils_tab_results @transform
  Scenario: Percent-change transformed DataFrame is non-empty
    When I transform a 10-row DataFrame with transformation "percent_change"
    Then the transformed DataFrame should not be empty

  @utils_tab_results @transform
  Scenario: Cumulative transformation adds _cum suffix columns
    When I transform a 10-row DataFrame with transformation "cumulative"
    Then the transformed DataFrame should contain columns with "_cum" suffix

  @utils_tab_results @transform
  Scenario: Cumulative columns are monotonically non-decreasing
    When I transform a 10-row DataFrame with transformation "cumulative"
    Then the cumulative columns should be monotonically non-decreasing
