Feature: Financial Enum types in utils_data_financial
  As a QWIM system developer
  I want the financial estimator type enums to expose well-defined member sets
  So that upstream callers can discover supported methods without hard-coding strings

  # -------------------------------------------------------------------------
  # expected_returns_estimator_type
  # -------------------------------------------------------------------------

  @utils @enums @smoke
  Scenario: expected_returns_estimator_type has exactly five members
    When I inspect the expected_returns_estimator_type enum
    Then the enum should have exactly 5 members

  @utils @enums
  Scenario: get_historical_methods returns at least one member
    When I call get_historical_methods on expected_returns_estimator_type
    Then the result should be a non-empty list

  @utils @enums
  Scenario: get_model_based_methods returns at least one member
    When I call get_model_based_methods on expected_returns_estimator_type
    Then the result should be a non-empty list

  @utils @enums
  Scenario: Historical and model-based groups for expected_returns are disjoint
    When I call get_historical_methods on expected_returns_estimator_type
    And I call get_model_based_methods on expected_returns_estimator_type
    Then the two result lists should have no common elements

  @utils @enums
  Scenario: All expected_returns members appear in at least one group
    When I retrieve all classmethod groups for expected_returns_estimator_type
    Then every enum member should appear in the union of all groups

  # -------------------------------------------------------------------------
  # prior_estimator_type
  # -------------------------------------------------------------------------

  @utils @enums @smoke
  Scenario: prior_estimator_type has exactly seven members
    When I inspect the prior_estimator_type enum
    Then the enum should have exactly 7 members

  @utils @enums
  Scenario: get_data_driven returns at least one prior method
    When I call get_data_driven on prior_estimator_type
    Then the result should be a non-empty list

  @utils @enums
  Scenario: get_equilibrium_based returns at least one prior method
    When I call get_equilibrium_based on prior_estimator_type
    Then the result should be a non-empty list

  @utils @enums
  Scenario: Data-driven and equilibrium-based prior groups are disjoint
    When I call get_data_driven on prior_estimator_type
    And I call get_equilibrium_based on prior_estimator_type
    Then the two result lists should have no common elements

  @utils @enums
  Scenario: All prior members appear in at least one group
    When I retrieve all classmethod groups for prior_estimator_type
    Then every enum member should appear in the union of all groups

  # -------------------------------------------------------------------------
  # distribution_estimator_type
  # -------------------------------------------------------------------------

  @utils @enums @smoke
  Scenario: distribution_estimator_type has exactly fourteen members
    When I inspect the distribution_estimator_type enum
    Then the enum should have exactly 14 members

  @utils @enums
  Scenario: get_univariate returns at least one distribution method
    When I call get_univariate on distribution_estimator_type
    Then the result should be a non-empty list

  @utils @enums
  Scenario: get_fat_tailed_distributions returns at least one distribution method
    When I call get_fat_tailed_distributions on distribution_estimator_type
    Then the result should be a non-empty list

  @utils @enums
  Scenario: All distribution members appear in at least one group
    When I retrieve all classmethod groups for distribution_estimator_type
    Then every enum member should appear in the union of all groups

  @utils @enums
  Scenario: get_simulation_ready contains members from other groups
    When I call get_simulation_ready on distribution_estimator_type
    Then the result should be a non-empty list
