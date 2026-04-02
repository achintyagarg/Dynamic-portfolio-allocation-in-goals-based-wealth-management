Feature: Day-count conventions for bond accrual calculations
  As a fixed-income analyst using QWIM
  I want accurate year-fraction calculations for all standard day-count conventions
  So that coupon and accrual amounts are correctly priced

  Background:
    Given day-count utility modules are importable

  # -------------------------------------------------------------------------
  # 30/360 convention
  # -------------------------------------------------------------------------

  @daycount @smoke @thirty_360
  Scenario: 30/360 year fraction for a 6-month period is approximately 0.5
    When I calculate the year fraction from "2024-01-01" to "2024-07-01" using "THIRTY_360"
    Then the year fraction should be approximately 0.5

  @daycount @thirty_360
  Scenario: 30/360 full calendar year from Jan to Jan equals 1.0
    When I calculate the year fraction from "2024-01-01" to "2025-01-01" using "THIRTY_360"
    Then the year fraction should be approximately 1.0

  @daycount @thirty_360
  Scenario: 30/360 year fraction for one month is approximately 0.08333
    When I calculate the year fraction from "2024-01-01" to "2024-02-01" using "THIRTY_360"
    Then the year fraction should be approximately 0.08333

  # -------------------------------------------------------------------------
  # ACT/360 convention
  # -------------------------------------------------------------------------

  @daycount @smoke @act_360
  Scenario: ACT/360 year fraction for 90 calendar days exceeds 0.25
    When I calculate the year fraction from "2024-01-01" to "2024-04-01" using "ACTUAL_360"
    Then the year fraction should be greater than 0.25

  @daycount @act_360
  Scenario: ACT/360 counts actual days divided by 360
    When I calculate the year fraction from "2024-01-01" to "2024-04-01" using "ACTUAL_360"
    Then the year fraction should be approximately 0.25278

  # -------------------------------------------------------------------------
  # ACT/365 convention
  # -------------------------------------------------------------------------

  @daycount @smoke @act_365
  Scenario: ACT/365 year fraction for a 6-month period is less than 0.5
    When I calculate the year fraction from "2024-01-01" to "2024-07-01" using "ACTUAL_365"
    Then the year fraction should be less than 0.5

  @daycount @act_365
  Scenario: ACT/365 for 181 days of 2024 is approximately 0.49589
    When I calculate the year fraction from "2024-01-01" to "2024-07-01" using "ACTUAL_365"
    Then the year fraction should be approximately 0.49589

  # -------------------------------------------------------------------------
  # ACT/ACT convention
  # -------------------------------------------------------------------------

  @daycount @smoke @act_act
  Scenario: ACT/ACT full year from Jan to Jan on a non-leap year equals 1.0
    When I calculate the year fraction from "2023-01-01" to "2024-01-01" using "ACTUAL_ACTUAL"
    Then the year fraction should be approximately 1.0

  @daycount @act_act
  Scenario: ACT/ACT full year on a leap year also equals 1.0
    When I calculate the year fraction from "2024-01-01" to "2025-01-01" using "ACTUAL_ACTUAL"
    Then the year fraction should be approximately 1.0

  @daycount @act_act
  Scenario: ACT/ACT half-year on a non-leap year is approximately 0.5
    When I calculate the year fraction from "2023-01-01" to "2023-07-02" using "ACTUAL_ACTUAL"
    Then the year fraction should be approximately 0.5

  # -------------------------------------------------------------------------
  # 30/365 convention
  # -------------------------------------------------------------------------

  @daycount @thirty_365
  Scenario: 30/365 full calendar year equals approximately 0.98630
    When I calculate the year fraction from "2024-01-01" to "2025-01-01" using "THIRTY_365"
    Then the year fraction should be approximately 0.98630

  # -------------------------------------------------------------------------
  # Factory / convention selection
  # -------------------------------------------------------------------------

  @daycount @factory @smoke
  Scenario Outline: Factory returns a working calculator for each convention
    When I create a daycount calculator for convention "<convention>"
    Then the calculator should return a positive year fraction for a 6-month period

    Examples:
      | convention      |
      | THIRTY_360      |
      | THIRTY_365      |
      | ACTUAL_360      |
      | ACTUAL_365      |
      | ACTUAL_ACTUAL   |

  # -------------------------------------------------------------------------
  # Additivity
  # -------------------------------------------------------------------------

  @daycount @additivity
  Scenario: ACT/ACT two half-years sum to the full year
    When I calculate the year fraction from "2023-01-01" to "2023-07-02" using "ACTUAL_ACTUAL"
    And I calculate the year fraction from "2023-07-02" to "2024-01-01" using "ACTUAL_ACTUAL"
    Then the sum of both year fractions should be approximately 1.0
