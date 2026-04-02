Feature: Portfolio QWIM construction and operations
  As a QWIM system user
  I want to create and manage financial portfolios
  So that I can track time-varying weights and calculate portfolio values

  # -------------------------------------------------------------------------
  # Construction
  # -------------------------------------------------------------------------

  @portfolio @construction @smoke
  Scenario: Create portfolio from component names
    When I create a portfolio named "Tech Portfolio" with components "AAPL,MSFT,GOOG"
    Then the portfolio name should be "Tech Portfolio"
    And the portfolio should have 3 components

  @portfolio @construction @smoke
  Scenario: Component names list is stored correctly
    When I create a portfolio named "Two-Fund" with components "VTI,BND"
    Then the component list should contain "VTI"
    And the component list should contain "BND"

  @portfolio @construction @smoke
  Scenario: Two portfolios created from different names are distinct objects
    When I create a portfolio named "Portfolio A" with components "VTI,AGG"
    And I create a second portfolio named "Portfolio B" with components "SPY,TLT"
    Then the two portfolios should be distinct objects

  # -------------------------------------------------------------------------
  # Properties
  # -------------------------------------------------------------------------

  @portfolio @properties
  Scenario: Portfolio weights DataFrame has correct column structure
    When I create a portfolio named "Simple" with components "VTI,AGG"
    Then the weights DataFrame should have a "Date" column
    And the weights DataFrame should have a "VTI" column

  @portfolio @validation
  Scenario: Portfolio equal weights sum to one
    When I create a portfolio named "Equal Weight" with components "A,B,C"
    Then the weights for each row should sum to approximately 1.0

  @portfolio @properties
  Scenario: Single-component portfolio has one component
    When I create a portfolio named "Single" with components "VTI"
    Then the portfolio should have 1 components

  # -------------------------------------------------------------------------
  # CSV loading
  # -------------------------------------------------------------------------

  @portfolio @csv @integration
  Scenario: Load portfolio from sample weights CSV file
    When I load a portfolio from the sample weights CSV file
    Then the portfolio should be created without error
    And the portfolio should have at least 1 component

  # -------------------------------------------------------------------------
  # Portfolio value calculation
  # -------------------------------------------------------------------------

  @portfolio @calculation @integration
  Scenario: Calculate portfolio values returns valid DataFrame
    When I load a portfolio from the sample weights CSV file
    And I load ETF price data from the sample ETF file
    And I calculate portfolio values with initial value 100.0
    Then the portfolio values DataFrame should not be empty
    And the first portfolio value should equal 100.0
    And all portfolio values should be positive

  # -------------------------------------------------------------------------
  # Benchmark
  # -------------------------------------------------------------------------

  @portfolio @benchmark @integration
  Scenario: Benchmark portfolio values differ from source
    When I load a portfolio from the sample weights CSV file
    And I load ETF price data from the sample ETF file
    And I calculate portfolio values with initial value 100.0
    And I create a benchmark from the portfolio values
    Then the benchmark values should differ from the source values
