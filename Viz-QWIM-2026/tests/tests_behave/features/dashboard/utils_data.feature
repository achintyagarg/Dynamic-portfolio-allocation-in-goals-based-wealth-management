Feature: Dashboard utils_data — raw data loading and portfolio validation
  As a dashboard developer
  I want reliable data-loading and validation utilities
  So that the dashboard only processes well-formed input data

  # -------------------------------------------------------------------------
  # get_input_data_raw
  # -------------------------------------------------------------------------

  @utils_data @raw_data @smoke
  Scenario: Raw input data loads without error
    When I load raw input data from the project directory
    Then the raw data should be loaded without error

  @utils_data @raw_data
  Scenario: Raw data contains Time_Series_Sample key
    When I load raw input data from the project directory
    Then the raw data should contain key "Time_Series_Sample"

  @utils_data @raw_data
  Scenario: Raw data contains Time_Series_ETFs key
    When I load raw input data from the project directory
    Then the raw data should contain key "Time_Series_ETFs"

  @utils_data @raw_data
  Scenario: Raw data contains Weights_My_Portfolio key
    When I load raw input data from the project directory
    Then the raw data should contain key "Weights_My_Portfolio"

  @utils_data @raw_data
  Scenario: Time_Series_Sample value is a non-empty DataFrame
    When I load raw input data from the project directory
    Then the value at key "Time_Series_Sample" should be a non-empty DataFrame

  @utils_data @raw_data
  Scenario: Time_Series_ETFs value is a non-empty DataFrame
    When I load raw input data from the project directory
    Then the value at key "Time_Series_ETFs" should be a non-empty DataFrame

  @utils_data @raw_data
  Scenario: Weights_My_Portfolio value is a non-empty DataFrame
    When I load raw input data from the project directory
    Then the value at key "Weights_My_Portfolio" should be a non-empty DataFrame

  # -------------------------------------------------------------------------
  # validate_portfolio_data
  # -------------------------------------------------------------------------

  @utils_data @validation @smoke
  Scenario: Valid two-column portfolio DataFrame passes validation
    When I validate a valid two-column portfolio DataFrame
    Then the validation result should be valid

  @utils_data @validation
  Scenario: Valid portfolio validation returns True as first element
    When I validate a valid two-column portfolio DataFrame
    Then the validation result first element should be True

  @utils_data @validation @edge_case
  Scenario: None portfolio data returns invalid validation result
    When I validate None as portfolio data
    Then the validation result should be invalid

  @utils_data @validation @edge_case
  Scenario: None portfolio data has a non-empty error message
    When I validate None as portfolio data
    Then the validation error message should not be empty

  @utils_data @validation @edge_case
  Scenario: DataFrame with wrong column name returns invalid validation result
    When I validate a DataFrame with wrong column names
    Then the validation result should be invalid

  @utils_data @validation @edge_case
  Scenario: Three-column DataFrame returns invalid validation result
    When I validate a DataFrame with an extra column
    Then the validation result should be invalid

  # -------------------------------------------------------------------------
  # calculate_portfolio_returns
  # -------------------------------------------------------------------------

  @utils_data @returns @smoke
  Scenario: Portfolio returns series is non-empty
    When I calculate portfolio returns from a valid portfolio DataFrame
    Then the returns series should not be empty
