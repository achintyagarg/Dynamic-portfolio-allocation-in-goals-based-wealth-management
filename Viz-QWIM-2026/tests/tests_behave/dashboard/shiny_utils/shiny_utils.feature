# Feature: Shiny Utils — QWIM Dashboard
#
# Validates the business-visible behaviour of utility helpers used across all
# dashboard tabs. Tests run against pure-Python module logic (no live Shiny
# server required).
#
# Run:
#     behave tests/tests_behave/dashboard/shiny_utils/
#
# Author:
#     QWIM Development Team
#
# Version:
#     0.1.0

Feature: Shiny utilities — validation, formatting, and error handling
  As a QWIM dashboard developer
  I want to verify that shared utility helpers enforce correct business rules
  So that all tabs relying on these utilities behave consistently and safely

  # ---------------------------------------------------------------------------
  # validate_portfolio_data
  # ---------------------------------------------------------------------------

  Background:
    Given the shiny_utils modules are importable

  Scenario: None portfolio data is rejected
    When I validate portfolio data with a None value
    Then validation returns False
    And the error message mentions "portfolio data"

  Scenario: Empty DataFrame is rejected
    When I validate an empty portfolio DataFrame
    Then validation returns False
    And the error message mentions "empty"

  Scenario: DataFrame missing the Date column is rejected
    When I validate a portfolio DataFrame without a Date column
    Then validation returns False
    And the error message mentions "Date"

  Scenario: Portfolio data missing Value column is rejected
    When I validate portfolio data without a Value column
    Then validation returns False
    And the error message mentions "Value"

  Scenario: Portfolio data with all non-numeric values is rejected
    When I validate portfolio data with all non-numeric Value entries
    Then validation returns False
    And the error message mentions "numeric"

  Scenario: Valid portfolio DataFrame is accepted
    When I validate a fully valid portfolio DataFrame
    Then validation returns True
    And the error message is empty

  Scenario: Valid weights DataFrame with components is accepted
    When I validate a weights DataFrame with component columns
    Then validation returns True
    And the error message is empty

  Scenario: Weights data without component columns is rejected
    When I validate weights data with only the Date column
    Then validation returns False
    And the error message mentions "component"

  # ---------------------------------------------------------------------------
  # format_currency_display
  # ---------------------------------------------------------------------------

  Scenario: Formatting zero produces "$0"
    When I format the amount 0 as currency
    Then the formatted result is "$0"

  Scenario: Formatting 1 000 000 produces "$1,000,000"
    When I format the amount 1000000 as currency
    Then the formatted result is "$1,000,000"

  Scenario: Formatting None produces "$0"
    When I format the amount None as currency
    Then the formatted result is "$0"

  # ---------------------------------------------------------------------------
  # validate_financial_amount
  # ---------------------------------------------------------------------------

  Scenario: Zero financial amount is valid
    When I validate the financial amount 0
    Then the validated amount is 0.0

  Scenario: Positive financial amount is valid
    When I validate the financial amount 250000
    Then the validated amount is 250000.0

  Scenario: Negative financial amount is rejected
    When I validate the financial amount -100
    Then a ValueError is raised mentioning "negative"

  Scenario: None financial amount becomes 0.0
    When I validate the financial amount None
    Then the validated amount is 0.0

  # ---------------------------------------------------------------------------
  # validate_age_range
  # ---------------------------------------------------------------------------

  Scenario: Age within valid range is accepted
    When I validate the age 45 within range 18 to 100
    Then the validated age is 45

  Scenario: Age below minimum is rejected
    When I validate the age 17 within range 18 to 100
    Then a ValueError is raised mentioning "between"

  Scenario: Age above maximum is rejected
    When I validate the age 101 within range 18 to 100
    Then a ValueError is raised mentioning "between"

  # ---------------------------------------------------------------------------
  # is_silent_exception
  # ---------------------------------------------------------------------------

  Scenario: AttributeError is treated as silent
    When I check if an AttributeError is a silent exception
    Then the result is True

  Scenario: ValueError is not treated as silent
    When I check if a ValueError is a silent exception
    Then the result is False

  # ---------------------------------------------------------------------------
  # Deprecated exception aliases emit DeprecationWarning
  # ---------------------------------------------------------------------------

  Scenario: Error_Silent_Initialization emits DeprecationWarning
    When I instantiate Error_Silent_Initialization
    Then a DeprecationWarning is emitted

  Scenario: Error_Dashboard_Initialization emits DeprecationWarning
    When I instantiate Error_Dashboard_Initialization
    Then a DeprecationWarning is emitted
