Feature: Dashboard formatting utilities — currency, percentage and number display
  As a QWIM dashboard user
  I want financial values displayed in a consistent, readable format
  So that reports and charts always show dollar-only currency and locale numbers

  # -------------------------------------------------------------------------
  # format_currency_value
  # -------------------------------------------------------------------------

  @formatting @currency @smoke
  Scenario: Formatted currency starts with a dollar sign
    When I format the currency value 1250.0
    Then the result should start with "$"

  @formatting @currency
  Scenario: Small currency amount formats to expected string
    When I format the currency value 500.0
    Then the currency result should equal "$500"

  @formatting @currency
  Scenario: Large currency amount formats to expected string
    When I format the currency value 1250000.0
    Then the currency result should equal "$1,250,000"

  @formatting @currency @edge_case
  Scenario: Zero formats to dollar zero
    When I format the currency value 0.0
    Then the currency result should equal "$0"

  @formatting @currency
  Scenario: Formatted currency result has no decimal point
    When I format the currency value 1250.0
    Then the result should not contain a decimal point

  @formatting @currency @edge_case
  Scenario: Fractional amount is rounded to nearest dollar
    When I format the currency value 999.9
    Then the currency result should equal "$1,000"

  @formatting @currency @edge_case
  Scenario: Negative currency amount starts with a dollar sign
    When I format the currency value -500.0
    Then the result should start with "$"

  # -------------------------------------------------------------------------
  # extract_numeric_from_currency_string
  # -------------------------------------------------------------------------

  @formatting @extract @smoke
  Scenario: Extract numeric value from small currency string
    When I extract the numeric value from "$500"
    Then the extracted value should equal 500.0

  @formatting @extract
  Scenario: Extract numeric value from large currency string
    When I extract the numeric value from "$1,250,000"
    Then the extracted value should equal 1250000.0

  @formatting @extract
  Scenario: Currency format and extract round-trip for small amount
    When I perform a currency round-trip for amount 500.0
    Then the round-trip should preserve the value

  @formatting @extract
  Scenario: Currency format and extract round-trip for large amount
    When I perform a currency round-trip for amount 1250000.0
    Then the round-trip should preserve the value

  @formatting @extract @edge_case
  Scenario: Extract from empty string returns zero
    When I extract the numeric value from ""
    Then the extracted value should equal 0.0

  @formatting @extract @edge_case
  Scenario: Extract from non-string input returns zero
    When I extract the numeric value from a non-string input
    Then the extracted value should equal 0.0

  # -------------------------------------------------------------------------
  # format_enhanced_percentage_display
  # -------------------------------------------------------------------------

  @formatting @percentage @smoke
  Scenario: Formatted percentage ends with percent sign
    When I format the percentage value 0.05
    Then the result should end with "%"

  @formatting @percentage
  Scenario: Five percent decimal formats to string containing 5
    When I format the percentage value 0.05
    Then the percentage result should contain "5"

  @formatting @percentage @edge_case
  Scenario: Zero decimal formats to zero percent
    When I format the percentage value 0.0
    Then the percentage result should contain "0"

  @formatting @percentage
  Scenario: Value already in percent form with multiply_by_hundred False
    When I format percentage value 15.5 without multiplying by hundred
    Then the percentage result should contain "15"

  # -------------------------------------------------------------------------
  # format_enhanced_number_display
  # -------------------------------------------------------------------------

  @formatting @number @smoke
  Scenario: Large number format contains a comma separator
    When I format the number value 1234567.0
    Then the result should contain a comma

  @formatting @number
  Scenario: Large number format result is not empty
    When I format the number value 1234567.0
    Then the number result should not be empty

  @formatting @number @edge_case
  Scenario: Zero number format result is not empty
    When I format the number value 0.0
    Then the number result should not be empty

  @formatting @number
  Scenario: Number with two decimal places result is not empty
    When I format the number value 1234.56 with 2 decimal places
    Then the number result should not be empty
