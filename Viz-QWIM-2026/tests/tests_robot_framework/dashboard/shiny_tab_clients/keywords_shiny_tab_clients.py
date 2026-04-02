"""Robot Framework keyword library for ``shiny_tab_clients`` tests.

Provides Python-level keywords that wrap the pure-Python logic from:

* ``src.dashboard.shiny_utils.utils_tab_clients``
* ``src.dashboard.shiny_tab_clients.*`` (source introspection)

No live Shiny server or browser is required.

Usage in .robot files::

    Library    ${CURDIR}${/}keywords_shiny_tab_clients.py

Author:
    QWIM Development Team

Version:
    0.2.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import inspect

from typing import Any

from robot.api.deco import keyword, library  # type: ignore[import-untyped]

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


_logger = get_logger(__name__)


@library(scope="SUITE", auto_keywords=False)
class keywords_shiny_tab_clients:
    """Robot Framework library for Clients Tab validation.

    All keywords operate on pure-Python module logic to avoid a live Shiny
    server dependency.

    Scope:
        SUITE - one library instance per Robot suite.
    """

    # ------------------------------------------------------------------
    # Library lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise lazy-import module caches."""
        self._utils: Any = None
        self._modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_utils(self) -> Any:
        """Return utils_tab_clients module (lazy import)."""
        if self._utils is None:
            try:
                import importlib

                self._utils = importlib.import_module(
                    "src.dashboard.shiny_utils.utils_tab_clients",
                )
                _logger.debug("utils_tab_clients imported")
            except ImportError as exc:
                raise AssertionError(f"utils_tab_clients not importable: {exc}") from exc
        return self._utils

    def _get_subtab(self, name: str) -> Any:
        """Return a shiny_tab_clients module by short name (lazy import)."""
        if name not in self._modules:
            try:
                import importlib

                self._modules[name] = importlib.import_module(
                    f"src.dashboard.shiny_tab_clients.{name}",
                )
                _logger.debug("Module '%s' imported", name)
            except ImportError as exc:
                raise AssertionError(f"Module '{name}' not importable: {exc}") from exc
        return self._modules[name]

    # ------------------------------------------------------------------
    # Currency formatting keywords
    # ------------------------------------------------------------------

    @keyword("Format Currency Display")
    def format_currency_display(self, amount: float | int | None) -> str:
        """Call format_currency_display and return the result.

        Args:
            amount: Numeric amount to format, or None.

        Returns:
            Formatted currency string, e.g. ``$1,234``.
        """
        utils = self._get_utils()
        result: str = utils.format_currency_display(amount)
        _logger.debug("format_currency_display(%s) = %s", amount, result)
        return result

    @keyword("Currency Display Should Be")
    def currency_display_should_be(self, amount: float | int | None, expected: str) -> None:
        """Assert that formatting *amount* produces *expected*.

        Args:
            amount:   Numeric amount to format.
            expected: Expected display string such as ``$1,234``.

        Raises:
            AssertionError: If the result does not match *expected*.
        """
        result = self.format_currency_display(amount)
        assert result == expected, (
            f"Currency display mismatch — expected '{expected}', got '{result}'"
        )
        _logger.info("Currency display '%s' verified for amount %s", expected, amount)

    @keyword("Currency Display Should Start With Dollar Sign")
    def currency_display_starts_with_dollar(self, amount: float | int) -> None:
        """Assert that the display result starts with ``$``.

        Args:
            amount: Numeric amount to format.

        Raises:
            AssertionError: If result does not start with ``$``.
        """
        result = self.format_currency_display(amount)
        assert result.startswith("$"), (
            f"Expected result starting with '$', got '{result}'"
        )

    # ------------------------------------------------------------------
    # Age validation keywords
    # ------------------------------------------------------------------

    @keyword("Validate Age Range")
    def validate_age_range(self, age: int, min_age: int = 18, max_age: int = 100) -> Any:
        """Call validate_age_range and return the result.

        Args:
            age:     Age value to validate.
            min_age: Minimum accepted age (default 18).
            max_age: Maximum accepted age (default 100).

        Returns:
            Validation result (True/int on success, False/None on failure).
        """
        utils = self._get_utils()
        return utils.validate_age_range(int(age), min_age=int(min_age), max_age=int(max_age))

    @keyword("Age Should Be Valid")
    def age_should_be_valid(self, age: int, min_age: int = 18, max_age: int = 100) -> None:
        """Assert that *age* passes validation within [min_age, max_age].

        Args:
            age:     Age value to validate.
            min_age: Minimum accepted age.
            max_age: Maximum accepted age.

        Raises:
            AssertionError: If validation fails or raises an exception.
        """
        try:
            result = self.validate_age_range(int(age), int(min_age), int(max_age))
            assert result is not False, (
                f"Age {age} failed validation (expected valid within [{min_age}, {max_age}])"
            )
            assert result is not None, (
                f"Age {age} failed validation (expected valid within [{min_age}, {max_age}])"
            )
        except Exception as exc:
            raise AssertionError(
                f"Age {age} raised unexpected exception: {exc}",
            ) from exc
        _logger.info("Age %d validated within [%d, %d]", age, min_age, max_age)

    @keyword("Age Should Be Invalid")
    def age_should_be_invalid(self, age: int, min_age: int = 18, max_age: int = 100) -> None:
        """Assert that *age* fails validation outside [min_age, max_age].

        Args:
            age:     Age value expected to fail.
            min_age: Range minimum.
            max_age: Range maximum.

        Raises:
            AssertionError: If the age unexpectedly passes validation.
        """
        try:
            result = self.validate_age_range(int(age), int(min_age), int(max_age))
            assert result is False or result is None, (
                f"Age {age} unexpectedly passed validation (range [{min_age}, {max_age}])"
            )
        except AssertionError:
            raise
        except Exception:  # noqa: BLE001, S110
            pass  # Raising is also acceptable for invalid input
        _logger.info("Age %d correctly rejected from [%d, %d]", age, min_age, max_age)

    # ------------------------------------------------------------------
    # Financial amount validation keywords
    # ------------------------------------------------------------------

    @keyword("Validate Financial Amount")
    def validate_financial_amount(self, amount: float) -> Any:
        """Call validate_financial_amount and return result.

        Args:
            amount: Monetary amount to validate.

        Returns:
            Validated amount (non-negative float or 0.0).
        """
        utils = self._get_utils()
        return utils.validate_financial_amount(float(amount))

    @keyword("Financial Amount Should Be Valid")
    def financial_amount_should_be_valid(self, amount: float) -> None:
        """Assert that *amount* passes financial amount validation.

        Args:
            amount: Monetary amount to validate.

        Raises:
            AssertionError: If validation fails or raises unexpectedly.
        """
        try:
            result = self.validate_financial_amount(float(amount))
            if result is not None:
                assert float(result) >= 0.0, (
                    f"Financial amount {amount} returned invalid result {result}"
                )
        except Exception as exc:
            raise AssertionError(
                f"Financial amount {amount} raised unexpected exception: {exc}",
            ) from exc
        _logger.info("Financial amount %s validated", amount)

    @keyword("Financial Amount Should Be Invalid")
    def financial_amount_should_be_invalid(self, amount: float) -> None:
        """Assert that *amount* fails validation (negative amounts).

        Args:
            amount: Monetary amount expected to fail.

        Raises:
            AssertionError: If the amount unexpectedly passes.
        """
        try:
            result = self.validate_financial_amount(float(amount))
            assert result is False or result is None, (
                f"Negative financial amount {amount} unexpectedly passed validation"
            )
        except AssertionError:
            raise
        except Exception:  # noqa: BLE001, S110
            pass  # Raising is acceptable
        _logger.info("Financial amount %s correctly rejected", amount)

    # ------------------------------------------------------------------
    # Source code inspection keywords
    # ------------------------------------------------------------------

    @keyword("Module Source Should Contain")
    def module_source_should_contain(self, module_name: str, text: str) -> None:
        """Assert that *text* appears in *module_name* source.

        Args:
            module_name: Short module name (e.g. ``subtab_personal_info``).
            text:        String that must appear in the module source.

        Raises:
            AssertionError: If *text* is not found in source.
        """
        mod = self._get_subtab(module_name)
        source = inspect.getsource(mod)
        assert text in source, (
            f"'{text}' not found in {module_name} source"
        )
        _logger.info("'%s' found in %s source", text, module_name)

    @keyword("Module Should Expose Callable")
    def module_should_expose_callable(self, module_name: str, func_name: str) -> None:
        """Assert that *module_name* exposes a callable named *func_name*.

        Args:
            module_name: Short module name.
            func_name:   Attribute name expected to be callable.

        Raises:
            AssertionError: If attribute missing or not callable.
        """
        mod = self._get_subtab(module_name)
        func = getattr(mod, func_name, None)
        assert func is not None, (
            f"'{func_name}' not found in module '{module_name}'"
        )
        assert callable(func), (
            f"'{func_name}' in module '{module_name}' is not callable"
        )
        _logger.info("'%s.%s' is callable", module_name, func_name)

    @keyword("Input ID Should Follow Convention")
    def input_id_should_follow_convention(self, module_name: str, input_id: str) -> None:
        """Assert that *input_id* appears in *module_name* source and starts with 'input_ID_'.

        Args:
            module_name: Short module name to search.
            input_id:    Complete input ID string to locate.

        Raises:
            AssertionError: If ID missing or does not follow convention.
        """
        assert input_id.startswith("input_ID_"), (
            f"Input ID '{input_id}' does not start with 'input_ID_'"
        )
        self.module_source_should_contain(module_name, input_id)
        _logger.info("Input ID convention verified: %s", input_id)

    @keyword("Default Value Should Be Present In Source")
    def default_value_should_be_in_source(self, module_name: str, default_value: str) -> None:
        """Assert that *default_value* appears in *module_name* source.

        Args:
            module_name:   Short module name to search.
            default_value: The literal default value string to find.

        Raises:
            AssertionError: If default value is not found.
        """
        self.module_source_should_contain(module_name, str(default_value))
        _logger.info("Default value '%s' verified in %s", default_value, module_name)

    @keyword("Get Module Source")
    def get_module_source(self, module_name: str) -> str:
        """Return source code of *module_name* as a string.

        Args:
            module_name: Short module name.

        Returns:
            Full source code as string.
        """
        mod = self._get_subtab(module_name)
        return inspect.getsource(mod)

    # ------------------------------------------------------------------
    # Constraint verification keywords
    # ------------------------------------------------------------------

    @keyword("Constraint Value Should Be Present")
    def constraint_value_should_be_in_source(
        self, module_name: str, constraint_value: int,
    ) -> None:
        """Assert that a numeric constraint appears in module source.

        Args:
            module_name:       Short module name to inspect.
            constraint_value:  Numeric bound that must appear in source.

        Raises:
            AssertionError: If the value is not found in source.
        """
        self.module_source_should_contain(module_name, str(int(constraint_value)))
        _logger.info("Constraint value %d verified in %s", constraint_value, module_name)
