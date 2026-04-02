Feature: Portfolio optimization enum utilities
  As a QWIM developer
  I want the portfolio_optimization_type and portfolio_optimization_feature_type enums
  to be fully functional after the aenum → stdlib enum migration
  So that all optimization wrappers dispatch correctly

  # =========================================================================
  # portfolio_optimization_type — members and classmethods
  # =========================================================================

  @portfolio_optimization @enum @smoke
  Scenario: portfolio_optimization_type has all 13 members
    When I inspect the portfolio_optimization_type enum
    Then the enum should have 13 members

  @portfolio_optimization @enum @smoke
  Scenario: portfolio_optimization_type values equal their names
    When I inspect the portfolio_optimization_type enum
    Then every member value should equal its name

  @portfolio_optimization @enum @smoke
  Scenario: get_basic_methods returns exactly 3 basic methods
    When I call get_basic_methods on portfolio_optimization_type
    Then I should get 3 methods
    And all methods should have names starting with "BASIC"

  @portfolio_optimization @enum @smoke
  Scenario: get_convex_methods returns exactly 5 convex methods
    When I call get_convex_methods on portfolio_optimization_type
    Then I should get 5 methods
    And all methods should have names starting with "CONVEX"

  @portfolio_optimization @enum @smoke
  Scenario: get_clustering_methods returns exactly 4 clustering methods
    When I call get_clustering_methods on portfolio_optimization_type
    Then I should get 4 methods
    And all methods should have names starting with "CLUSTERING"

  @portfolio_optimization @enum @smoke
  Scenario: get_ensemble_methods returns exactly 1 ensemble method
    When I call get_ensemble_methods on portfolio_optimization_type
    Then I should get 1 method
    And all methods should have names starting with "ENSEMBLE"

  @portfolio_optimization @enum @regression
  Scenario: All four method groups together cover all 13 members
    When I combine basic, convex, clustering, and ensemble methods
    Then the combined set should equal all 13 portfolio_optimization_type members

  @portfolio_optimization @enum @regression
  Scenario: portfolio_optimization_type subscript lookup works (stdlib Enum)
    When I access portfolio_optimization_type by name "BASIC_EQUAL_WEIGHTED"
    Then I should get the BASIC_EQUAL_WEIGHTED member

  # =========================================================================
  # portfolio_optimization_feature_type — members and classmethods
  # =========================================================================

  @portfolio_optimization @feature_enum @smoke
  Scenario: portfolio_optimization_feature_type has all 24 members
    When I inspect the portfolio_optimization_feature_type enum
    Then the feature enum should have 24 members

  @portfolio_optimization @feature_enum @smoke
  Scenario: get_objectives returns exactly 4 objective features
    When I call get_objectives on portfolio_optimization_feature_type
    Then I should get 4 features

  @portfolio_optimization @feature_enum @smoke
  Scenario: get_all_constraints returns exactly 12 constraint features
    When I call get_all_constraints on portfolio_optimization_feature_type
    Then I should get 12 features
    And the result should include CONSTRAINTS_CUSTOM

  @portfolio_optimization @feature_enum @smoke
  Scenario: get_convex_features returns exactly 13 convex-preserving features
    When I call get_convex_features on portfolio_optimization_feature_type
    Then I should get 13 features

  @portfolio_optimization @feature_enum @regression
  Scenario: Integer features are not in convex features (breaks convexity)
    When I get the integer features
    And I get the convex features
    Then integer features and convex features should be disjoint

  # =========================================================================
  # End-to-end: enum drives skfolio optimization dispatching
  # =========================================================================

  @portfolio_optimization @e2e @smoke
  Scenario: Equal-weighted optimization produces valid portfolio
    Given I have synthetic daily-returns data for 3 assets and 100 days
    When I run calc_skfolio_optimization_basic with "BASIC_EQUAL_WEIGHTED"
    Then I should receive a portfolio_QWIM object
    And the portfolio should have 3 components
    And all weights should sum to approximately 1.0

  @portfolio_optimization @e2e @regression
  Scenario: Enum instance and string produce identical equal-weighted portfolios
    Given I have synthetic daily-returns data for 3 assets and 100 days
    When I run basic optimization with the enum member BASIC_EQUAL_WEIGHTED
    And I run basic optimization with the string "BASIC_EQUAL_WEIGHTED"
    Then the two portfolios should have identical weights

  @portfolio_optimization @e2e @regression
  Scenario: Convex optimization runs without UnboundLocalError (benchmark_pandas fix)
    Given I have synthetic daily-returns data for 5 assets and 252 days
    When I run calc_skfolio_optimization_convex with "CONVEX_RISK_BUDGETING"
    Then I should receive a portfolio_QWIM object
    And all weights should sum to approximately 1.0
