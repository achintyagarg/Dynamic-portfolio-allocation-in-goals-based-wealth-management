# Feature: Results Tab — QWIM Dashboard
#
# Validates the business-visible behaviour of the Results tab, including
# module importability, input/output ID naming conventions, security
# constants, and the filename sanitization helper.
#
# Tests run against pure-Python module logic (no live Shiny server required).
#
# Run (from project root):
#     behave tests/tests_behave/features/ --tags=shiny_tab_results
#
# Author:
#     QWIM Development Team
#
# Version:
#     0.1.0

Feature: Results Tab — module API, naming conventions, and security helpers
  As a QWIM dashboard developer
  I want to verify that the Results tab enforces strict naming conventions,
  exports correct module-level functions, and rejects unsafe PDF filenames
  So that the dashboard is consistent and user-generated filenames are safe

  # ---------------------------------------------------------------------------
  # Background — shared module availability
  # ---------------------------------------------------------------------------

  Background:
    Given the shiny_tab_results modules are importable

  # ---------------------------------------------------------------------------
  # Module structure
  # ---------------------------------------------------------------------------

  @shiny_tab_results @module @smoke
  Scenario: tab_results UI function is callable
    When I check the tab_results module exports
    Then "tab_results_ui" should be a callable

  @shiny_tab_results @module @smoke
  Scenario: tab_results server function is callable
    When I check the tab_results module exports
    Then "tab_results_server" should be a callable

  @shiny_tab_results @module
  Scenario: subtab_reporting UI and server are callable
    When I check the subtab_reporting module exports
    Then "subtab_reporting_ui" should be a callable
    And "subtab_reporting_server" should be a callable

  @shiny_tab_results @module
  Scenario: subtab_simulation UI and server are callable
    When I check the subtab_simulation module exports
    Then "subtab_simulation_ui" should be a callable
    And "subtab_simulation_server" should be a callable

  # ---------------------------------------------------------------------------
  # Security constants
  # ---------------------------------------------------------------------------

  @shiny_tab_results @security @smoke
  Scenario: MAX_FILENAME_LENGTH is a positive integer
    When I inspect the subtab_reporting security constants
    Then MAX_FILENAME_LENGTH should be a positive integer

  @shiny_tab_results @security
  Scenario: FORBIDDEN_FILENAME_PARTS includes path traversal
    When I inspect the subtab_reporting security constants
    Then FORBIDDEN_FILENAME_PARTS should contain ".."

  @shiny_tab_results @security
  Scenario: FORBIDDEN_FILENAME_PARTS includes backslash
    When I inspect the subtab_reporting security constants
    Then FORBIDDEN_FILENAME_PARTS should contain "\\"

  @shiny_tab_results @security
  Scenario: ALLOWED_FILENAME_PATTERN is a compiled regular expression
    When I inspect the subtab_reporting security constants
    Then ALLOWED_FILENAME_PATTERN should be a compiled regex

  # ---------------------------------------------------------------------------
  # sanitize_filename_for_security — acceptance
  # ---------------------------------------------------------------------------

  @shiny_tab_results @sanitize @smoke
  Scenario: A well-formed PDF filename is accepted
    When I sanitize the results filename "QWIM_Report_2024.pdf"
    Then the results sanitization result should indicate "valid"
    And the results sanitized filename should not be empty

  @shiny_tab_results @sanitize @smoke
  Scenario: A filename with spaces is accepted
    When I sanitize the results filename "Annual Report 2024.pdf"
    Then the results sanitization result should indicate "valid"

  @shiny_tab_results @sanitize
  Scenario: The return value is always a 3-tuple
    When I sanitize the results filename "Test.pdf"
    Then the results sanitization return should be a tuple with 3 elements

  # ---------------------------------------------------------------------------
  # sanitize_filename_for_security — rejection
  # ---------------------------------------------------------------------------

  @shiny_tab_results @sanitize @smoke
  Scenario: An empty filename is rejected
    When I sanitize a results empty filename
    Then the results sanitization result should indicate "invalid"
    And the results sanitized filename should be empty

  @shiny_tab_results @sanitize @smoke
  Scenario: A filename with path traversal is rejected
    When I sanitize the results filename "../../etc/passwd.pdf"
    Then the results sanitization result should indicate "invalid"

  @shiny_tab_results @sanitize
  Scenario: A filename without .pdf extension is rejected
    When I sanitize the results filename "Report_2024.docx"
    Then the results sanitization result should indicate "invalid"

  @shiny_tab_results @sanitize
  Scenario: An extremely long filename is rejected
    When I sanitize a results filename that is 300 characters long
    Then the results sanitization result should indicate "invalid"

  @shiny_tab_results @sanitize
  Scenario: A filename starting with a dot is rejected
    When I sanitize the results filename ".hidden_file.pdf"
    Then the results sanitization result should indicate "invalid"

  @shiny_tab_results @sanitize @parametrized
  Scenario Outline: Forbidden special characters are always rejected
    When I sanitize the results filename "<test_filename>"
    Then the results sanitization result should indicate "invalid"

    Examples:
      | test_filename        |
      | Report$2024.pdf      |
      | Report%2024.pdf      |
      | Report~2024.pdf      |
      | Report*2024.pdf      |

  # ---------------------------------------------------------------------------
  # Simulation constants
  # ---------------------------------------------------------------------------

  @shiny_tab_results @simulation @smoke
  Scenario: ALL_ETF_SYMBOLS contains exactly 12 entries
    When I inspect the subtab_simulation constants
    Then ALL_ETF_SYMBOLS should have 12 entries

  @shiny_tab_results @simulation
  Scenario: DEFAULT_SELECTED_ETFS are all in ALL_ETF_SYMBOLS
    When I inspect the subtab_simulation constants
    Then every DEFAULT_SELECTED_ETFS entry should be in ALL_ETF_SYMBOLS

  @shiny_tab_results @simulation
  Scenario: DISTRIBUTION_CHOICES covers normal, lognormal, and student_t
    When I inspect the subtab_simulation constants
    Then DISTRIBUTION_CHOICES should include keys "normal", "lognormal", and "student_t"

  @shiny_tab_results @simulation
  Scenario: RNG_TYPE_CHOICES covers pcg64 and mt19937
    When I inspect the subtab_simulation constants
    Then RNG_TYPE_CHOICES should include keys "pcg64" and "mt19937"
