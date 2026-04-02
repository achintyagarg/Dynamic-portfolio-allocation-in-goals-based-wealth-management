"""Robot Framework keyword library for ``shiny_tab_results`` tests.

Provides Python-level keywords that wrap the pure-Python logic from:

* ``src.dashboard.shiny_tab_results.*`` (module introspection)
* ``src.dashboard.shiny_tab_results.subtab_reporting`` (security helpers)
* ``src.dashboard.shiny_tab_results.subtab_simulation`` (simulation constants)

No live Shiny server or browser is required.

Usage in .robot files::

    Library    ${CURDIR}${/}keywords_shiny_tab_results.py

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import io
import re
import sys

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Robot Framework replaces sys.stderr with StringIO, which lacks .buffer.
# Some Shiny transitive imports call sys.stderr.buffer; add a shim so they
# don't crash before the first keyword runs.
# ---------------------------------------------------------------------------
if not hasattr(sys.stderr, "buffer"):
    sys.stderr.buffer = io.BytesIO()  # type: ignore[attr-defined]

from robot.api.deco import keyword, library  # type: ignore[import-untyped]

try:
    from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger
    _logger = get_logger(__name__)
except Exception:
    import logging
    _logger = logging.getLogger(__name__)


@library(scope="SUITE", auto_keywords=False)
class keywords_shiny_tab_results:
    """Robot Framework library for Results Tab validation.

    All keywords operate on pure-Python module logic; no live Shiny server
    or browser is required.

    Scope:
        SUITE - one library instance per Robot suite.
    """

    # ------------------------------------------------------------------
    # Library lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise lazy-import module caches."""
        self._modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_module(self, name: str) -> Any:
        """Return a shiny_tab_results submodule by short name (lazy import)."""
        # Robot Framework replaces sys.stderr with StringIO per keyword call;
        # re-apply buffer shim so transitive Shiny imports don't crash.
        if not hasattr(sys.stderr, "buffer"):
            sys.stderr.buffer = io.BytesIO()  # type: ignore[attr-defined]
        if name not in self._modules:
            try:
                self._modules[name] = importlib.import_module(
                    f"src.dashboard.shiny_tab_results.{name}",
                )
                _logger.debug(f"Module '{name}' imported")
            except ImportError as exc:
                raise AssertionError(f"Module '{name}' not importable: {exc}") from exc
        return self._modules[name]

    def _get_source(self, name: str) -> str:
        """Return the source text of a shiny_tab_results submodule."""
        # Re-apply buffer shim (find_spec imports parent packages).
        if not hasattr(sys.stderr, "buffer"):
            sys.stderr.buffer = io.BytesIO()  # type: ignore[attr-defined]
        spec = importlib.util.find_spec(f"src.dashboard.shiny_tab_results.{name}")
        if spec and spec.origin:
            return Path(spec.origin).read_text(encoding="utf-8")
        return ""

    # ------------------------------------------------------------------
    # Smoke — module API surface
    # ------------------------------------------------------------------

    @keyword("Module Should Expose Callable")
    def module_should_expose_callable(self, module_name: str, func_name: str) -> None:
        """Assert that ``module_name`` exposes a callable attribute named ``func_name``.

        Args:
            module_name: Short module name (e.g. ``tab_results``).
            func_name: Name of the expected callable attribute.
        """
        mod = self._get_module(module_name)
        assert hasattr(mod, func_name), (
            f"Module '{module_name}' does not have attribute '{func_name}'"
        )
        fn = getattr(mod, func_name)
        assert callable(fn), (
            f"'{module_name}.{func_name}' exists but is not callable"
        )
        _logger.info(f"'{module_name}.{func_name}' is callable — OK")

    @keyword("Module Should Have Attribute")
    def module_should_have_attribute(self, module_name: str, attr_name: str) -> None:
        """Assert that ``module_name`` has the attribute ``attr_name``.

        Args:
            module_name: Short module name.
            attr_name: Expected attribute name.
        """
        mod = self._get_module(module_name)
        assert hasattr(mod, attr_name), (
            f"Module '{module_name}' does not have attribute '{attr_name}'"
        )
        _logger.info(f"'{module_name}.{attr_name}' exists — OK")

    # ------------------------------------------------------------------
    # Naming — input / output ID conventions
    # ------------------------------------------------------------------

    @keyword("Input IDs Should Follow Naming Convention")
    def input_ids_should_follow_naming_convention(
        self, module_name: str, expected_prefix: str,
    ) -> None:
        """Assert all input_ID_ strings in the module follow the expected prefix.

        Args:
            module_name: Short module name.
            expected_prefix: Required prefix for all input IDs.
        """
        source = self._get_source(module_name)
        pattern = r'["\']?(input_ID_tab_results[^"\'>\s,)]+)["\']?'
        all_ids = list(dict.fromkeys(re.findall(pattern, source)))

        if not all_ids:
            _logger.warning(f"No input IDs found in '{module_name}'")
            return

        bad_ids = [id_ for id_ in all_ids if not id_.startswith(expected_prefix)]
        assert not bad_ids, (
            f"Input IDs in '{module_name}' do not start with '{expected_prefix}': {bad_ids}"
        )
        _logger.info(
            f"All {len(all_ids)} input IDs in '{module_name}' follow convention — OK",
        )

    @keyword("Server Function Should Accept Parameter")
    def server_function_should_accept_parameter(
        self, module_name: str, func_name: str, param_name: str,
    ) -> None:
        """Assert that a server function accepts a required parameter.

        Args:
            module_name: Short module name.
            func_name: Name of the server function.
            param_name: Expected parameter name.
        """
        mod = self._get_module(module_name)
        fn = getattr(mod, func_name)
        sig = inspect.signature(fn)
        assert param_name in sig.parameters, (
            f"'{module_name}.{func_name}' does not have parameter '{param_name}'"
        )
        _logger.info(f"'{func_name}' has '{param_name}' parameter — OK")

    # ------------------------------------------------------------------
    # Security constants — subtab_reporting
    # ------------------------------------------------------------------

    @keyword("Security Constant Should Be Positive Integer")
    def security_constant_should_be_positive_integer(
        self, module_name: str, const_name: str,
    ) -> None:
        """Assert a module-level constant is a positive integer.

        Args:
            module_name: Short module name.
            const_name: Name of the constant attribute.
        """
        mod = self._get_module(module_name)
        val = getattr(mod, const_name)
        assert isinstance(val, int), (
            f"'{module_name}.{const_name}' is {type(val)}, expected int"
        )
        assert val > 0, (
            f"'{module_name}.{const_name}' is {val}, expected positive int"
        )
        _logger.info(f"'{module_name}.{const_name}' = {val} (positive int) — OK")

    @keyword("List Constant Should Contain Value")
    def list_constant_should_contain_value(
        self, module_name: str, const_name: str, expected_value: str,
    ) -> None:
        """Assert a module-level list/tuple constant contains a given value.

        Args:
            module_name: Short module name.
            const_name: Name of the list constant.
            expected_value: Value that must be present.
        """
        mod = self._get_module(module_name)
        container = getattr(mod, const_name)
        assert expected_value in container, (
            f"'{module_name}.{const_name}' does not contain '{expected_value}'"
        )
        _logger.info(
            f"'{module_name}.{const_name}' contains '{expected_value}' — OK",
        )

    @keyword("Attribute Should Be Compiled Regex")
    def attribute_should_be_compiled_regex(
        self, module_name: str, attr_name: str,
    ) -> None:
        """Assert a module-level attribute is a compiled ``re.Pattern``.

        Args:
            module_name: Short module name.
            attr_name: Name of the attribute.
        """
        mod = self._get_module(module_name)
        val = getattr(mod, attr_name)
        assert isinstance(val, re.Pattern), (
            f"'{module_name}.{attr_name}' is {type(val)}, expected re.Pattern"
        )
        _logger.info(f"'{module_name}.{attr_name}' is a compiled regex — OK")

    # ------------------------------------------------------------------
    # sanitize_filename_for_security
    # ------------------------------------------------------------------

    @keyword("Sanitize Filename Should Return Tuple")
    def sanitize_filename_should_return_tuple(
        self, module_name: str, filename: str,
    ) -> None:
        """Assert sanitize_filename_for_security returns a 3-element tuple.

        Args:
            module_name: Short module name (``subtab_reporting``).
            filename: Filename string to sanitize.
        """
        mod = self._get_module(module_name)
        fn = getattr(mod, "sanitize_filename_for_security")
        result = fn(filename)
        assert isinstance(result, tuple), (
            f"sanitize_filename_for_security returned {type(result)}, expected tuple"
        )
        assert len(result) == 3, (
            f"sanitize_filename_for_security returned {len(result)}-element tuple, expected 3"
        )
        _logger.info(f"sanitize_filename_for_security('{filename}') returned 3-tuple — OK")

    @keyword("Filename Should Be Accepted")
    def filename_should_be_accepted(self, module_name: str, filename: str) -> None:
        """Assert sanitize_filename_for_security accepts the given filename.

        Args:
            module_name: Short module name.
            filename: Filename string expected to be valid.
        """
        mod = self._get_module(module_name)
        fn = getattr(mod, "sanitize_filename_for_security")
        is_valid, sanitized, message = fn(filename)
        assert is_valid is True, (
            f"Expected '{filename}' to be accepted, but it was rejected: {message}"
        )
        assert len(sanitized) > 0, (
            f"Accepted filename '{filename}' but sanitized result is empty"
        )
        _logger.info(f"Filename '{filename}' accepted — OK")

    @keyword("Filename Should Be Rejected")
    def filename_should_be_rejected(self, module_name: str, filename: str) -> None:
        """Assert sanitize_filename_for_security rejects the given filename.

        Args:
            module_name: Short module name.
            filename: Filename string expected to be invalid.
        """
        mod = self._get_module(module_name)
        fn = getattr(mod, "sanitize_filename_for_security")
        is_valid, sanitized, message = fn(filename)
        assert is_valid is False, (
            f"Expected '{filename}' to be rejected, but it was accepted"
        )
        assert sanitized == "", (
            f"Rejected filename '{filename}' but sanitized result is not empty: '{sanitized}'"
        )
        _logger.info(f"Filename '{filename}' rejected — OK ({message})")

    # ------------------------------------------------------------------
    # Simulation constants — subtab_simulation
    # ------------------------------------------------------------------

    @keyword("Sequence Constant Should Have Length")
    def sequence_constant_should_have_length(
        self, module_name: str, const_name: str, expected_length: int,
    ) -> None:
        """Assert a module-level sequence constant has the expected length.

        Args:
            module_name: Short module name.
            const_name: Name of the sequence constant.
            expected_length: Expected number of elements.
        """
        mod = self._get_module(module_name)
        seq = getattr(mod, const_name)
        assert len(seq) == expected_length, (
            f"'{module_name}.{const_name}' has {len(seq)} entries, expected {expected_length}"
        )
        _logger.info(
            f"'{module_name}.{const_name}' length={expected_length} — OK",
        )

    @keyword("All Items In Constant Should Be In Other Constant")
    def all_items_in_constant_should_be_in_other_constant(
        self, module_name: str, source_const: str, container_const: str,
    ) -> None:
        """Assert every item in ``source_const`` is present in ``container_const``.

        Args:
            module_name: Short module name.
            source_const: Name of the constant whose items are checked.
            container_const: Name of the constant that must contain all items.
        """
        mod = self._get_module(module_name)
        source = getattr(mod, source_const)
        container = getattr(mod, container_const)
        missing = [item for item in source if item not in container]
        assert not missing, (
            f"Items in '{source_const}' missing from '{container_const}': {missing}"
        )
        _logger.info(
            f"All items in '{source_const}' are in '{container_const}' — OK",
        )

    @keyword("Dict Constant Should Contain Key")
    def dict_constant_should_contain_key(
        self, module_name: str, const_name: str, expected_key: str,
    ) -> None:
        """Assert a module-level dict constant contains a given key.

        Args:
            module_name: Short module name.
            const_name: Name of the dict constant.
            expected_key: Key that must be present.
        """
        mod = self._get_module(module_name)
        d = getattr(mod, const_name)
        assert expected_key in d, (
            f"'{module_name}.{const_name}' does not contain key '{expected_key}'"
        )
        _logger.info(
            f"'{module_name}.{const_name}' contains key '{expected_key}' — OK",
        )

    @keyword("Dict Constant Should Have Length")
    def dict_constant_should_have_length(
        self, module_name: str, const_name: str, expected_length: int,
    ) -> None:
        """Assert a module-level dict constant has the expected number of entries.

        Args:
            module_name: Short module name.
            const_name: Name of the dict constant.
            expected_length: Expected number of entries.
        """
        mod = self._get_module(module_name)
        d = getattr(mod, const_name)
        assert len(d) == expected_length, (
            f"'{module_name}.{const_name}' has {len(d)} entries, expected {expected_length}"
        )
        _logger.info(
            f"'{module_name}.{const_name}' length={expected_length} — OK",
        )
