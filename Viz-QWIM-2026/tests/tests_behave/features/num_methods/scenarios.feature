Feature: Scenario generation from distributions and CMA
  As a QWIM quantitative analyst
  I want to generate financial return scenarios from parametric distributions
  So that I can stress-test portfolios and project forward-looking risk

  # -------------------------------------------------------------------------
  # Scenarios_Distribution — construction
  # -------------------------------------------------------------------------

  @scenarios @construction @smoke
  Scenario: Normal distribution scenario created without error
    When I create a Normal distribution scenario with 3 components and 60 days
    Then the scenario object should be created without error

  @scenarios @construction @smoke
  Scenario: Student-t distribution scenario created without error
    When I create a Student-t distribution scenario with 3 components and 60 days
    Then the scenario object should be created without error

  @scenarios @construction @smoke
  Scenario: Scenarios_CMA created without error
    When I create a CMA scenario with 3 asset classes and 60 days
    Then the scenario object should be created without error

  # -------------------------------------------------------------------------
  # Scenarios_Distribution.generate() — shape
  # -------------------------------------------------------------------------

  @scenarios @generation @smoke
  Scenario: Normal scenario generate returns Polars DataFrame
    When I create a Normal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then the generated DataFrame should be a Polars DataFrame

  @scenarios @generation
  Scenario: Normal scenario generate returns correct row count
    When I create a Normal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then the generated DataFrame should have 60 rows

  @scenarios @generation
  Scenario: Normal scenario generate has Date column
    When I create a Normal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then the generated DataFrame should have a "Date" column

  @scenarios @generation
  Scenario: Normal scenario generate has Float64 component columns
    When I create a Normal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then all component columns should be of type Float64

  @scenarios @generation
  Scenario: Normal scenario generate contains no null values
    When I create a Normal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then the generated DataFrame should contain no null values

  # -------------------------------------------------------------------------
  # Scenarios_CMA.generate() — shape
  # -------------------------------------------------------------------------

  @scenarios @generation @cma
  Scenario: CMA scenario generate returns correct row count
    When I create a CMA scenario with 3 asset classes and 60 days
    And I call generate on the scenario
    Then the generated DataFrame should have 60 rows

  @scenarios @generation @cma
  Scenario: CMA scenario has no null values
    When I create a CMA scenario with 3 asset classes and 60 days
    And I call generate on the scenario
    Then the generated DataFrame should contain no null values

  # -------------------------------------------------------------------------
  # Reproducibility
  # -------------------------------------------------------------------------

  @scenarios @reproducibility
  Scenario: Same seed produces identical Normal scenario output
    When I create two Normal distribution scenarios with the same seed
    Then the two generated DataFrames should be identical

  @scenarios @reproducibility
  Scenario: Different seeds produce different Normal scenario output
    When I create two Normal distribution scenarios with different seeds
    Then the two generated DataFrames should differ

  # -------------------------------------------------------------------------
  # Base-class utilities
  # -------------------------------------------------------------------------

  @scenarios @utilities
  Scenario: validate_scenarios passes after generate
    When I create a Normal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then validate_scenarios should pass without error

  @scenarios @utilities
  Scenario: get_component_series returns correct length
    When I create a Normal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then get_component_series for the first component should have 60 rows

  @scenarios @utilities
  Scenario: get_returns_matrix returns correct shape
    When I create a Normal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then get_returns_matrix should have shape 60 by 3

  @scenarios @utilities
  Scenario: filter_by_date_range reduces row count
    When I create a Normal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then filtering to half the date range should reduce the row count

  # -------------------------------------------------------------------------
  # Lognormal positivity
  # -------------------------------------------------------------------------

  @scenarios @lognormal
  Scenario: Lognormal scenario all values are positive
    When I create a Lognormal distribution scenario with 3 components and 60 days
    And I call generate on the scenario
    Then all component values in the generated DataFrame should be positive
