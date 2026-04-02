Feature: Reporting package public API
  As a QWIM dashboard developer
  I want the reporting package to export exactly its three documented public symbols
  So that consumers of the reporting API get a clean, predictable interface

  Background:
    Given the reporting package is importable

  # -------------------------------------------------------------------------
  # Export surface: positive checks
  # -------------------------------------------------------------------------

  @reporting @api @smoke
  Scenario: compile_typst_report is exported from the reporting package
    Then "compile_typst_report" should be importable from "src.dashboard.reporting"
    And "compile_typst_report" should be callable

  @reporting @api @smoke
  Scenario: generate_report_PDF is exported from the reporting package
    Then "generate_report_PDF" should be importable from "src.dashboard.reporting"
    And "generate_report_PDF" should be callable

  @reporting @api @smoke
  Scenario: validate_polars_DF is exported from the reporting package
    Then "validate_polars_DF" should be importable from "src.dashboard.reporting"
    And "validate_polars_DF" should be callable

  # -------------------------------------------------------------------------
  # Export surface: regression guard — removed symbol
  # -------------------------------------------------------------------------

  @reporting @api @regression
  Scenario: build_typst_data_context is NOT exported from the reporting package
    Then "build_typst_data_context" should NOT be accessible from the reporting package
    And the reporting package __all__ should not contain "build_typst_data_context"

  # -------------------------------------------------------------------------
  # __all__ completeness
  # -------------------------------------------------------------------------

  @reporting @api @regression
  Scenario: The reporting package __all__ contains exactly three symbols
    Then the reporting package __all__ should contain exactly 3 entries
    And the reporting package __all__ should contain "compile_typst_report"
    And the reporting package __all__ should contain "generate_report_PDF"
    And the reporting package __all__ should contain "validate_polars_DF"

  # -------------------------------------------------------------------------
  # validate_polars_DF functional
  # -------------------------------------------------------------------------

  @reporting @functional @smoke
  Scenario: validate_polars_DF accepts a well-formed Polars DataFrame
    Given a Polars DataFrame with columns Date and Value
    When I call validate_polars_DF with the DataFrame
    Then the validation result should not be False

  @reporting @functional @edge_case
  Scenario: validate_polars_DF handles None input without unhandled exception
    Given a None input value
    When I call validate_polars_DF with the None input
    Then no unhandled exception should have been raised
