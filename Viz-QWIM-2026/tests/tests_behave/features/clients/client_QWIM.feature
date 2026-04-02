Feature: Client QWIM construction and data management
  As a QWIM advisor
  I want to construct and manage client profiles
  So that I can store personal information, assets, goals and income

  # -------------------------------------------------------------------------
  # Construction
  # -------------------------------------------------------------------------

  @client @construction @smoke
  Scenario: Create a primary client
    When I create a primary client with id "CLI_001" first name "Jane" last name "Smith"
    Then the client should be created without error

  @client @construction @smoke
  Scenario: Create a partner client
    When I create a partner client with id "CLI_002" first name "John" last name "Smith"
    Then the client should be created without error

  @client @construction @smoke
  Scenario: Two distinct clients are different objects
    When I create a primary client with id "CLI_001" first name "Jane" last name "Smith"
    And I create a partner client with id "CLI_002" first name "John" last name "Smith"
    Then the two clients should be different objects

  # -------------------------------------------------------------------------
  # Personal information
  # -------------------------------------------------------------------------

  @client @personal_info @smoke
  Scenario: Update personal information returns True
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update personal info with current age 45 retirement age 65 and risk tolerance 5
    Then the personal info update should return True

  @client @personal_info
  Scenario: Current age is stored correctly after update
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update personal info with current age 45 retirement age 65 and risk tolerance 5
    Then the stored current age should equal 45

  @client @personal_info
  Scenario: Retirement age is stored correctly after update
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update personal info with current age 45 retirement age 65 and risk tolerance 5
    Then the stored retirement age should equal 65

  @client @personal_info
  Scenario: Risk tolerance is stored correctly after update
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update personal info with current age 45 retirement age 65 and risk tolerance 5
    Then the stored risk tolerance should equal 5

  @client @personal_info
  Scenario: Retirement age is greater than current age after update
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update personal info with current age 45 retirement age 65 and risk tolerance 5
    Then the retirement age should be greater than the current age

  @client @personal_info
  Scenario: Personal info DataFrame is non-empty after update
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update personal info with current age 45 retirement age 65 and risk tolerance 5
    Then the personal info DataFrame should not be empty

  # -------------------------------------------------------------------------
  # Assets
  # -------------------------------------------------------------------------

  @client @assets @smoke
  Scenario: Update assets returns True
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update assets with retirement account 500000 and stocks 250000
    Then the assets update should return True

  @client @assets
  Scenario: Total assets equal sum of individual categories
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update assets with retirement account 500000 and stocks 250000
    Then the total assets should equal 750000.0

  @client @assets
  Scenario: Total assets are positive after update
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update assets with retirement account 500000 and stocks 250000
    Then the total assets should be positive

  # -------------------------------------------------------------------------
  # Goals
  # -------------------------------------------------------------------------

  @client @goals @smoke
  Scenario: Update goals returns True
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update goals with a retirement goal of 2000000.0 by year 2044
    Then the goals update should return True

  @client @goals
  Scenario: Goals DataFrame is non-empty after update
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update goals with a retirement goal of 2000000.0 by year 2044
    Then the goals DataFrame should not be empty

  # -------------------------------------------------------------------------
  # Income
  # -------------------------------------------------------------------------

  @client @income @smoke
  Scenario: Update income returns True
    Given a primary client with id "CLI_001" first name "Jane" last name "Smith"
    When I update income with salary 120000 and social security 30000
    Then the income update should return True
