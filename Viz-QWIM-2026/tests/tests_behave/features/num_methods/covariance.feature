Feature: Covariance and correlation matrix estimation
  As a QWIM quantitative analyst
  I want to estimate covariance and correlation matrices from return data
  So that I can use them for portfolio optimisation and risk analysis

  # -------------------------------------------------------------------------
  # covariance_estimator enum
  # -------------------------------------------------------------------------

  @covariance @construction @smoke
  Scenario: Empirical covariance matrix created from valid DataFrame
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    Then the covariance matrix object should be created without error

  @covariance @construction @smoke
  Scenario: LedoitWolf covariance matrix created from valid DataFrame
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "LEDOIT_WOLF" estimator
    Then the covariance matrix object should be created without error

  @covariance @construction @smoke
  Scenario: OAS covariance matrix created from valid DataFrame
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "ORACLE_APPROXIMATING_SHRINKAGE" estimator
    Then the covariance matrix object should be created without error

  # -------------------------------------------------------------------------
  # Shape and structure
  # -------------------------------------------------------------------------

  @covariance @properties
  Scenario: Covariance matrix has correct shape for 3 components
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    Then the covariance matrix should have shape 3 by 3

  @covariance @properties
  Scenario: Covariance matrix has correct component count
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    Then the covariance matrix should report 3 components

  @covariance @properties
  Scenario: Covariance matrix has correct observation count
    Given a returns DataFrame with 60 observations and 3 components
    When I build a covariance matrix using the "EMPIRICAL" estimator
    Then the covariance matrix should report 60 observations

  # -------------------------------------------------------------------------
  # Validation
  # -------------------------------------------------------------------------

  @covariance @validation
  Scenario: Empirical covariance matrix passes validation
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    Then the validate_covariance_matrix method should pass without error

  @covariance @validation
  Scenario: Covariance matrix diagonal is all positive
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    Then all diagonal elements of the covariance matrix should be positive

  @covariance @validation
  Scenario: Covariance matrix is symmetric
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    Then the covariance matrix should be symmetric

  # -------------------------------------------------------------------------
  # Correlation conversion
  # -------------------------------------------------------------------------

  @covariance @correlation
  Scenario: Correlation matrix diagonal equals 1.0
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    And I compute the correlation matrix
    Then all diagonal elements of the correlation matrix should equal 1.0

  @covariance @correlation
  Scenario: Correlation values are in the range minus one to one
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    And I compute the correlation matrix
    Then all correlation values should be between -1.0 and 1.0

  @covariance @correlation
  Scenario: Correlation matrix is symmetric
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    And I compute the correlation matrix
    Then the correlation matrix should be symmetric

  # -------------------------------------------------------------------------
  # Component accessors
  # -------------------------------------------------------------------------

  @covariance @accessor
  Scenario: Component variance equals diagonal of full matrix
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    Then the variance accessor for component "AAPL" should match the matrix diagonal

  @covariance @accessor
  Scenario: Component covariance matches off-diagonal elements
    Given a returns DataFrame with 3 components and 60 observations
    When I build a covariance matrix using the "EMPIRICAL" estimator
    Then the covariance between "AAPL" and "MSFT" should match the matrix value

  # -------------------------------------------------------------------------
  # distance_estimator_type
  # -------------------------------------------------------------------------

  @covariance @distance_estimator
  Scenario: get_correlation_based returns a list
    When I call get_correlation_based on distance_estimator_type
    Then the distance estimator result should be a non-empty list

  @covariance @distance_estimator
  Scenario: get_rank_based returns a list
    When I call get_rank_based on distance_estimator_type
    Then the distance estimator result should be a non-empty list

  @covariance @distance_estimator
  Scenario: All distance estimator members are categorised
    When I collect all category members of distance_estimator_type
    Then all distance_estimator_type members should be covered
