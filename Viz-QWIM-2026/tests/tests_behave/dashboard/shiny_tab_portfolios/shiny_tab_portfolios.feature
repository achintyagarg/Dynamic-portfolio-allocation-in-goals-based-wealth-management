# Feature: Portfolios Tab — QWIM Dashboard
#
# Validates the business-visible behaviour of the Portfolios tab, including
# input ID naming conventions, optimization method constants, default values,
# and module API contracts.
#
# Tests run against pure-Python module logic (no live Shiny server required).
#
# Run:
#     behave tests/tests_behave/dashboard/shiny_tab_portfolios/
#
# Author:
#     QWIM Development Team
#
# Version:
#     0.2.0

Feature: Portfolios Tab — input naming conventions and optimization constants
  As a QWIM dashboard developer
  I want to verify that the Portfolios tab enforces strict naming conventions,
  correct optimization method constants, and proper module API contracts
  So that the dashboard is consistent and all optimization methods are accessible

  # ---------------------------------------------------------------------------
  # Naming convention
  # ---------------------------------------------------------------------------

  Background:
    Given the shiny_tab_portfolios modules are importable

  Scenario: All portfolio analysis input IDs follow the hierarchical convention
    Given the subtab_portfolios_analysis source code is loaded
    When I check all portfolio analysis input identifiers
    Then each identifier starts with "input_ID_tab_portfolios_subtab_portfolios_analysis_"

  Scenario: All portfolio comparison input IDs follow the hierarchical convention
    Given the subtab_portfolios_comparison source code is loaded
    When I check all portfolio comparison input identifiers
    Then each identifier starts with "input_ID_tab_portfolios_subtab_portfolios_comparison_"

  Scenario: All weights analysis input IDs follow the hierarchical convention
    Given the subtab_weights_analysis source code is loaded
    When I check all weights analysis input identifiers
    Then each identifier starts with "input_ID_tab_portfolios_subtab_weights_analysis_"

  Scenario: All skfolio input IDs follow the hierarchical convention
    Given the subtab_portfolios_skfolio source code is loaded
    When I check all skfolio input identifiers
    Then each identifier starts with "input_ID_tab_portfolios_subtab_skfolio_"

  # ---------------------------------------------------------------------------
  # Default values
  # ---------------------------------------------------------------------------

  Scenario: Portfolio analysis time period defaults to "1y"
    Given the subtab_portfolios_analysis source code is loaded
    When I inspect the default time period value
    Then the default value is "1y"

  Scenario: Portfolio analysis type defaults to "returns"
    Given the subtab_portfolios_analysis source code is loaded
    When I inspect the default analysis type value
    Then the default value is "returns"

  Scenario: skfolio time period defaults to "3y"
    Given the subtab_portfolios_skfolio source code is loaded
    When I inspect the skfolio default time period value
    Then the default value is "3y"

  Scenario: skfolio method1 category defaults to "basic"
    Given the subtab_portfolios_skfolio source code is loaded
    When I inspect the skfolio method1 default category value
    Then the default value is "basic"

  Scenario: skfolio method2 category defaults to "convex"
    Given the subtab_portfolios_skfolio source code is loaded
    When I inspect the skfolio method2 default category value
    Then the default value is "convex"

  # ---------------------------------------------------------------------------
  # Optimization method constants
  # ---------------------------------------------------------------------------

  Scenario: OPTIMIZATION_CATEGORIES contains exactly four entries
    Given the subtab_portfolios_skfolio source code is loaded
    When I count the optimization categories
    Then the count is 4

  Scenario: BASIC_METHODS contains exactly three entries
    Given the subtab_portfolios_skfolio source code is loaded
    When I count the basic optimization methods
    Then the count is 3

  Scenario: CONVEX_METHODS contains exactly five entries
    Given the subtab_portfolios_skfolio source code is loaded
    When I count the convex optimization methods
    Then the count is 5

  Scenario: CLUSTERING_METHODS contains exactly four entries
    Given the subtab_portfolios_skfolio source code is loaded
    When I count the clustering optimization methods
    Then the count is 4

  Scenario: ENSEMBLE_METHODS contains exactly one entry
    Given the subtab_portfolios_skfolio source code is loaded
    When I count the ensemble optimization methods
    Then the count is 1

  Scenario: OBJECTIVE_FUNCTIONS contains exactly four entries
    Given the subtab_portfolios_skfolio source code is loaded
    When I count the objective functions
    Then the count is 4

  Scenario: Total optimization methods across all categories is 13
    Given the subtab_portfolios_skfolio source code is loaded
    When I count all optimization methods across all categories
    Then the count is 13

  Scenario: No method key is duplicated across categories
    Given the subtab_portfolios_skfolio source code is loaded
    When I collect all method keys from all category dicts
    Then no duplicate keys exist

  # ---------------------------------------------------------------------------
  # Module API
  # ---------------------------------------------------------------------------

  Scenario: subtab_portfolios_analysis exposes callable UI and server
    Given the subtab_portfolios_analysis source code is loaded
    When I inspect the module API for subtab_portfolios_analysis
    Then the module exposes a callable "subtab_portfolios_analysis_ui"
    And the module exposes a callable "subtab_portfolios_analysis_server"

  Scenario: subtab_portfolios_comparison exposes callable UI and server
    Given the subtab_portfolios_comparison source code is loaded
    When I inspect the module API for subtab_portfolios_comparison
    Then the module exposes a callable "subtab_portfolios_comparison_ui"
    And the module exposes a callable "subtab_portfolios_comparison_server"

  Scenario: subtab_weights_analysis exposes callable UI and server
    Given the subtab_weights_analysis source code is loaded
    When I inspect the module API for subtab_weights_analysis
    Then the module exposes a callable "subtab_weights_analysis_ui"
    And the module exposes a callable "subtab_weights_analysis_server"

  Scenario: subtab_portfolios_skfolio exposes callable UI and server
    Given the subtab_portfolios_skfolio source code is loaded
    When I inspect the module API for subtab_portfolios_skfolio
    Then the module exposes a callable "subtab_portfolios_skfolio_ui"
    And the module exposes a callable "subtab_portfolios_skfolio_server"

  Scenario: tab_portfolios exposes callable UI and server
    Given the subtab_portfolios_analysis source code is loaded
    When I inspect the module API for tab_portfolios
    Then the module exposes a callable "tab_portfolios_ui"
    And the module exposes a callable "tab_portfolios_server"

  # ---------------------------------------------------------------------------
  # Weights analysis chart types
  # ---------------------------------------------------------------------------

  Scenario: Weights analysis supports stacked area chart
    Given the subtab_weights_analysis source code is loaded
    When I check supported chart types
    Then "stacked_area" is among the chart types

  Scenario: Weights analysis supports heatmap chart
    Given the subtab_weights_analysis source code is loaded
    When I check supported chart types
    Then "heatmap" is among the chart types

  # ---------------------------------------------------------------------------
  # OUTPUT_DIR configuration
  # ---------------------------------------------------------------------------

  Scenario: All portfolio modules define OUTPUT_DIR as a Path
    Given the shiny_tab_portfolios modules are importable
    When I inspect OUTPUT_DIR across all portfolio modules
    Then each OUTPUT_DIR is a Path object
