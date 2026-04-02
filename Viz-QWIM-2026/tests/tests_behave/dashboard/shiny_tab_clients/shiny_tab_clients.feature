# Feature: Clients Tab — QWIM Dashboard
#
# Validates the business-visible behaviour of the Clients tab, including
# input ID naming conventions, default values, constraint enforcement, and
# currency/age formatting rules.
#
# Tests run against pure-Python module logic (no live Shiny server required).
#
# Run:
#     behave tests/tests_behave/dashboard/shiny_tab_clients/
#
# Author:
#     QWIM Development Team
#
# Version:
#     0.2.0

Feature: Clients Tab — input validation and naming conventions
  As a QWIM dashboard developer
  I want to verify that the Clients tab enforces strict naming conventions,
  sensible defaults, and correct business constraints
  So that the dashboard is consistent and provides safe input handling

  # ---------------------------------------------------------------------------
  # Naming convention
  # ---------------------------------------------------------------------------

  Background:
    Given the shiny_tab_clients modules are importable

  Scenario: All primary personal-info input IDs follow the hierarchical convention
    Given the subtab_personal_info source code is loaded
    When I check all primary client input identifiers
    Then each identifier starts with "input_ID_tab_clients_subtab_clients_personal_info_client_primary"

  Scenario: All partner personal-info input IDs follow the hierarchical convention
    Given the subtab_personal_info source code is loaded
    When I check all partner client input identifiers
    Then each identifier starts with "input_ID_tab_clients_subtab_clients_personal_info_client_partner"

  Scenario: All asset input IDs follow the hierarchical convention
    Given the subtab_assets source code is loaded
    When I check primary client asset input identifiers
    Then each identifier starts with "input_ID_tab_clients_subtab_clients_assets_client_primary"

  Scenario: All income input IDs follow the hierarchical convention
    Given the subtab_income source code is loaded
    When I check primary client income input identifiers
    Then each identifier starts with "input_ID_tab_clients_subtab_clients_income_client_primary"

  Scenario: All goal input IDs follow the hierarchical convention
    Given the subtab_goals source code is loaded
    When I check primary client goal input identifiers
    Then each identifier starts with "input_ID_tab_clients_subtab_clients_goals_client_primary"

  # ---------------------------------------------------------------------------
  # Default values
  # ---------------------------------------------------------------------------

  Scenario: Primary client name defaults to "Anne Smith"
    Given the subtab_personal_info source code is loaded
    When I inspect the default value for the primary client name
    Then the default value is "Anne Smith"

  Scenario: Partner client name defaults to "William Smith"
    Given the subtab_personal_info source code is loaded
    When I inspect the default value for the partner client name
    Then the default value is "William Smith"

  Scenario: Primary client current age defaults to 60
    Given the subtab_personal_info source code is loaded
    When I inspect the default value for the primary client current age
    Then the default value is 60

  Scenario: Marital status defaults to "married"
    Given the subtab_personal_info source code is loaded
    When I inspect the default value for the primary client marital status
    Then the default value is "married"

  Scenario: Risk tolerance defaults to "moderate"
    Given the subtab_personal_info source code is loaded
    When I inspect the default value for the primary client risk tolerance
    Then the default value is "moderate"

  Scenario: Primary client gender defaults to "female"
    Given the subtab_personal_info source code is loaded
    When I inspect the default value for the primary client gender
    Then the default value is "female"

  Scenario: Partner client gender defaults to "male"
    Given the subtab_personal_info source code is loaded
    When I inspect the default value for the partner client gender
    Then the default value is "male"

  Scenario: ZIP code defaults to 12345
    Given the subtab_personal_info source code is loaded
    When I inspect the default value for the primary client ZIP code
    Then the default value is 12345

  # ---------------------------------------------------------------------------
  # Choice sets
  # ---------------------------------------------------------------------------

  Scenario: Marital status choices include all expected options
    Given the subtab_personal_info source code is loaded
    When I inspect the marital status choice set
    Then the choices include "single"
    And the choices include "married"
    And the choices include "divorced"
    And the choices include "widowed"
    And the choices include "separated"
    And the choices include "domestic_partnership"

  Scenario: Risk tolerance choices include all five tiers
    Given the subtab_personal_info source code is loaded
    When I inspect the risk tolerance choice set
    Then the choices include "conservative"
    And the choices include "moderate_conservative"
    And the choices include "moderate"
    And the choices include "moderate_aggressive"
    And the choices include "aggressive"

  # ---------------------------------------------------------------------------
  # Constraint bounds
  # ---------------------------------------------------------------------------

  Scenario: Current-age input is bounded between 18 and 100
    Given the subtab_personal_info source code is loaded
    When I check the bounds for the primary current age input
    Then the minimum value is 18
    And the maximum value is 100

  Scenario: Retirement-age input is bounded between 50 and 80
    Given the subtab_personal_info source code is loaded
    When I check the bounds for the primary retirement age input
    Then the minimum value is 50
    And the maximum value is 80

  Scenario: Asset values are constrained below 100 million
    Given the subtab_assets source code is loaded
    When I check the asset constraint upper bound
    Then the upper bound is 100000000

  Scenario: Income values are constrained below 5 million
    Given the subtab_income source code is loaded
    When I check the income constraint upper bound
    Then the upper bound is 5000000

  # ---------------------------------------------------------------------------
  # Currency formatting integration
  # ---------------------------------------------------------------------------

  Scenario Outline: Currency display formatting rounds to whole dollars
    Given the utils_tab_clients module is importable
    When I format the amount <amount> as currency
    Then the display result is "<expected>"

    Examples:
      | amount    | expected   |
      | 0         | $0         |
      | 1000      | $1,000     |
      | 100000    | $100,000   |
      | 1234.56   | $1,235     |
      | 5000000   | $5,000,000 |

  Scenario: Formatting None returns "$0"
    Given the utils_tab_clients module is importable
    When I format None as currency
    Then the display result is "$0"

  # ---------------------------------------------------------------------------
  # Age validation integration
  # ---------------------------------------------------------------------------

  Scenario Outline: validate_age_range accepts ages within bounds
    Given the utils_tab_clients module is importable
    When I validate age <age> with min=18 and max=100
    Then the validation passes

    Examples:
      | age |
      | 18  |
      | 45  |
      | 65  |
      | 100 |

  Scenario Outline: validate_age_range rejects ages outside bounds
    Given the utils_tab_clients module is importable
    When I validate age <age> with min=18 and max=100
    Then the validation fails

    Examples:
      | age |
      | 5   |
      | 150 |

  # ---------------------------------------------------------------------------
  # Financial amount validation integration
  # ---------------------------------------------------------------------------

  Scenario Outline: Non-negative financial amounts pass validation
    Given the utils_tab_clients module is importable
    When I validate the financial amount <amount>
    Then the validation passes

    Examples:
      | amount    |
      | 0         |
      | 500       |
      | 100000    |
      | 5000000   |

  Scenario: Negative financial amounts are rejected
    Given the utils_tab_clients module is importable
    When I validate the financial amount -1
    Then the validation fails

  # ---------------------------------------------------------------------------
  # tab_clients orchestration
  # ---------------------------------------------------------------------------

  Scenario: tab_clients_server references all five subtab server return keys
    Given the tab_clients source code is loaded
    When I inspect the server return dictionary keys
    Then the key "clients_Personal_Info_Server" is present
    And the key "clients_Assets_Server" is present
    And the key "clients_Goals_Server" is present
    And the key "clients_Income_Server" is present
    And the key "clients_Summary_Server" is present

  Scenario: tab_clients_ui defines the top-level navset_tab ID
    Given the tab_clients source code is loaded
    When I inspect the UI navset definition
    Then the navset ID "ID_tab_clients_tabs_all" is defined

  # ---------------------------------------------------------------------------
  # Module API surface
  # ---------------------------------------------------------------------------

  Scenario: All six subtab modules expose callable UI functions
    Given the shiny_tab_clients modules are importable
    When I inspect each subtab UI function
    Then every UI function is callable

  Scenario: All six subtab modules expose callable server functions
    Given the shiny_tab_clients modules are importable
    When I inspect each subtab server function
    Then every server function is callable
