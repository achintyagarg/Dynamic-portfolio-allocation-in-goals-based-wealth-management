Feature: Simulation Standard Monte Carlo engine
  As a QWIM analyst
  I want to run Monte Carlo simulations with different return distributions
  So that I can model retirement portfolio outcomes under uncertainty

  # -------------------------------------------------------------------------
  # Construction
  # -------------------------------------------------------------------------

  @simulation @construction @smoke
  Scenario: Create Normal distribution simulation with two components
    When I create a Normal simulation with components "VTI,AGG"
    Then the simulation should be created without error

  @simulation @construction @smoke
  Scenario: Create Lognormal distribution simulation with two components
    When I create a Lognormal simulation with components "VTI,AGG"
    Then the simulation should be created without error

  @simulation @construction @smoke
  Scenario: Create Student-T distribution simulation with two components
    When I create a Student-T simulation with components "VTI,AGG"
    Then the simulation should be created without error

  @simulation @construction
  Scenario: Create Normal simulation with three components
    When I create a Normal simulation with components "VTI,AGG,VNQ"
    Then the simulation should be created without error

  # -------------------------------------------------------------------------
  # run() output validity
  # -------------------------------------------------------------------------

  @simulation @calculation @smoke
  Scenario: Normal simulation run returns valid DataFrame
    When I create a Normal simulation with components "VTI,AGG"
    And I run the simulation
    Then the simulation results should be a valid non-empty DataFrame

  @simulation @calculation
  Scenario: Normal simulation results contain a Date column
    When I create a Normal simulation with components "VTI,AGG"
    And I run the simulation
    Then the simulation results should contain a "Date" column

  @simulation @calculation
  Scenario: Normal simulation results have correct row count
    When I create a Normal simulation with components "VTI,AGG"
    And I run the simulation
    Then the simulation results should have 30 rows

  @simulation @calculation
  Scenario: Normal simulation results have correct column count
    When I create a Normal simulation with components "VTI,AGG"
    And I run the simulation
    Then the simulation results should have 51 columns

  @simulation @validation
  Scenario: All Normal simulation portfolio values are positive
    When I create a Normal simulation with components "VTI,AGG"
    And I run the simulation
    Then all simulation portfolio values should be positive

  @simulation @validation
  Scenario: Normal simulation first row values are near the initial value
    When I create a Normal simulation with components "VTI,AGG"
    And I run the simulation
    Then each scenario column first value should be within 15 percent of 100.0

  @simulation @calculation
  Scenario: Normal simulation Date column is sorted ascending
    When I create a Normal simulation with components "VTI,AGG"
    And I run the simulation
    Then the simulation Date column should be sorted ascending

  # -------------------------------------------------------------------------
  # Lognormal / Student-T smoke
  # -------------------------------------------------------------------------

  @simulation @calculation
  Scenario: Lognormal simulation run returns valid DataFrame
    When I create a Lognormal simulation with components "VTI,AGG"
    And I run the simulation
    Then the simulation results should be a valid non-empty DataFrame

  @simulation @validation
  Scenario: All Lognormal simulation portfolio values are positive
    When I create a Lognormal simulation with components "VTI,AGG"
    And I run the simulation
    Then all simulation portfolio values should be positive

  @simulation @validation
  Scenario: Lognormal first row values are near the initial value
    When I create a Lognormal simulation with components "VTI,AGG"
    And I run the simulation
    Then each scenario column first value should be within 15 percent of 100.0

  @simulation @calculation
  Scenario: Student-T simulation run returns valid DataFrame
    When I create a Student-T simulation with components "VTI,AGG"
    And I run the simulation
    Then the simulation results should be a valid non-empty DataFrame

  @simulation @validation
  Scenario: Student-T first row values are near the initial value
    When I create a Student-T simulation with components "VTI,AGG"
    And I run the simulation
    Then each scenario column first value should be within 15 percent of 100.0

  # -------------------------------------------------------------------------
  # Reproducibility
  # -------------------------------------------------------------------------

  @simulation @reproducibility @smoke
  Scenario: Same seed produces identical simulation results
    When I create two identical Normal simulations with components "VTI,AGG" and the same seed
    And I run both simulations
    Then the two result DataFrames should be identical

  @simulation @reproducibility
  Scenario: Different seeds produce valid but potentially different results
    When I create two Normal simulations with components "VTI,AGG" using seed 42 and seed 99
    And I run both simulations
    Then both simulation result DataFrames should be valid
