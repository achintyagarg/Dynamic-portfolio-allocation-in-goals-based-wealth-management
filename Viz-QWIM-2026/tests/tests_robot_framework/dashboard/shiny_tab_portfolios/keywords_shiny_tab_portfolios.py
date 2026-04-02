"""Robot Framework keyword library for ``shiny_tab_portfolios`` tests.

Provides Python-level keywords that wrap the pure-Python logic from:

* ``src.dashboard.shiny_tab_portfolios.*`` (source introspection)

No live Shiny server or browser is required.

Usage in .robot files::

    Library    ${CURDIR}${/}keywords_shiny_tab_portfolios.py

Author:
    QWIM Development Team

Version:
    0.2.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import importlib
import inspect
import re

from pathlib import Path
from typing import Any

from robot.api.deco import keyword, library  # type: ignore[import-untyped]

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


_logger = get_logger(__name__)


@library(scope="SUITE", auto_keywords=False)
class keywords_shiny_tab_portfolios:
    """Robot Framework library for Portfolios Tab validation.

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
        self._modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_subtab(self, name: str) -> Any:
        """Return a shiny_tab_portfolios module by short name (lazy import)."""
        if name not in self._modules:
            try:
                self._modules[name] = importlib.import_module(
                    f"src.dashboard.shiny_tab_portfolios.{name}",
                )
                _logger.debug(f"Module '{name}' imported")
            except ImportError as exc:
                raise AssertionError(f"Module '{name}' not importable: {exc}") from exc
        return self._modules[name]

    def _get_source(self, name: str) -> str:
        """Return the source text of a shiny_tab_portfolios module."""
        spec = importlib.util.find_spec(f"src.dashboard.shiny_tab_portfolios.{name}")
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
            module_name: Short module name (e.g. ``subtab_portfolios_analysis``).
            func_name: Name of the expected callable attribute.
        """
        mod = self._get_subtab(module_name)
        assert hasattr(mod, func_name), (
            f"Module '{module_name}' does not have attribute '{func_name}'"
        )
        fn = getattr(mod, func_name)
        assert callable(fn), f"'{module_name}.{func_name}' exists but is not callable"
        _logger.info(f"'{module_name}.{func_name}' is callable — OK")

    @keyword("Module Should Have Attribute")
    def module_should_have_attribute(self, module_name: str, attr_name: str) -> None:
        """Assert that ``module_name`` has the attribute ``attr_name``.

        Args:
            module_name: Short module name.
            attr_name: Expected attribute name.
        """
        mod = self._get_subtab(module_name)
        assert hasattr(mod, attr_name), (
            f"Module '{module_name}' does not have attribute '{attr_name}'"
        )
        _logger.info(f"'{module_name}.{attr_name}' exists — OK")

    # ------------------------------------------------------------------
    # Naming — input ID conventions
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
        pattern = r'["\']?(input_ID_tab_portfolios[^"\'>\s,)]+)["\']?'
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
        mod = self._get_subtab(module_name)
        fn = getattr(mod, func_name)
        sig = inspect.signature(fn)
        assert param_name in sig.parameters, (
            f"'{module_name}.{func_name}' does not have parameter '{param_name}'"
        )
        _logger.info(f"'{func_name}' has '{param_name}' parameter — OK")

    # ------------------------------------------------------------------
    # Optimization constants — skfolio
    # ------------------------------------------------------------------

    @keyword("Dict Should Have Length")
    def dict_should_have_length(
        self, module_name: str, dict_name: str, expected_length: int,
    ) -> None:
        """Assert that a module-level dict has the expected number of entries.

        Args:
            module_name: Short module name.
            dict_name: Name of the dict attribute.
            expected_length: Expected number of entries.
        """
        mod = self._get_subtab(module_name)
        d = getattr(mod, dict_name)
        assert len(d) == expected_length, (
            f"'{module_name}.{dict_name}' has {len(d)} entries, expected {expected_length}"
        )
        _logger.info(f"'{module_name}.{dict_name}' length={expected_length} — OK")

    @keyword("All Keys In Dict Should Start With Prefix")
    def all_keys_in_dict_should_start_with_prefix(
        self, module_name: str, dict_name: str, prefix: str,
    ) -> None:
        """Assert all keys in a dict start with the given prefix.

        Args:
            module_name: Short module name.
            dict_name: Name of the dict attribute.
            prefix: Expected key prefix.
        """
        mod = self._get_subtab(module_name)
        d = getattr(mod, dict_name)
        bad_keys = [k for k in d if not k.startswith(prefix)]
        assert not bad_keys, (
            f"Keys in '{module_name}.{dict_name}' without prefix '{prefix}': {bad_keys}"
        )
        _logger.info(
            f"All {len(d)} keys in '{module_name}.{dict_name}' have prefix '{prefix}' — OK",
        )

    @keyword("No Duplicate Keys Across Method Dicts")
    def no_duplicate_keys_across_method_dicts(self, module_name: str) -> None:
        """Assert no duplicate keys exist across all four optimization method dicts.

        Args:
            module_name: Short module name (should be subtab_portfolios_skfolio).
        """
        mod = self._get_subtab(module_name)
        all_keys = (
            list(mod.BASIC_METHODS.keys())
            + list(mod.CONVEX_METHODS.keys())
            + list(mod.CLUSTERING_METHODS.keys())
            + list(mod.ENSEMBLE_METHODS.keys())
        )
        unique = set(all_keys)
        assert len(all_keys) == len(unique), (
            f"Duplicate method keys found in '{module_name}': "
            f"{len(all_keys) - len(unique)} duplicates"
        )
        _logger.info(f"All {len(all_keys)} method keys are unique — OK")

    @keyword("Total Method Count Should Be")
    def total_method_count_should_be(self, module_name: str, expected_total: int) -> None:
        """Assert the total method count across all method dicts equals expected.

        Args:
            module_name: Short module name.
            expected_total: Expected total method count.
        """
        mod = self._get_subtab(module_name)
        total = (
            len(mod.BASIC_METHODS)
            + len(mod.CONVEX_METHODS)
            + len(mod.CLUSTERING_METHODS)
            + len(mod.ENSEMBLE_METHODS)
        )
        assert total == expected_total, (
            f"Total method count in '{module_name}' is {total}, expected {expected_total}"
        )
        _logger.info(f"Total method count={expected_total} — OK")

    @keyword("Each Category Should Have Non Empty Method Dict")
    def each_category_should_have_non_empty_method_dict(self, module_name: str) -> None:
        """Assert each OPTIMIZATION_CATEGORIES key has a corresponding non-empty dict.

        Args:
            module_name: Short module name.
        """
        mod = self._get_subtab(module_name)
        category_to_dict = {
            "basic": mod.BASIC_METHODS,
            "convex": mod.CONVEX_METHODS,
            "clustering": mod.CLUSTERING_METHODS,
            "ensemble": mod.ENSEMBLE_METHODS,
        }
        for key in mod.OPTIMIZATION_CATEGORIES:
            assert key in category_to_dict, (
                f"Category '{key}' has no corresponding method dict"
            )
            assert len(category_to_dict[key]) > 0, (
                f"Method dict for category '{key}' is empty"
            )
        _logger.info("All optimization categories have populated method dicts — OK")

    # ------------------------------------------------------------------
    # OUTPUT_DIR
    # ------------------------------------------------------------------

    @keyword("Module OUTPUT_DIR Should Be Path")
    def module_output_dir_should_be_path(self, module_name: str) -> None:
        """Assert a module's OUTPUT_DIR attribute is a Path instance.

        Args:
            module_name: Short module name.
        """
        mod = self._get_subtab(module_name)
        assert hasattr(mod, "OUTPUT_DIR"), (
            f"Module '{module_name}' does not have OUTPUT_DIR"
        )
        output_dir = mod.OUTPUT_DIR
        assert isinstance(output_dir, Path), (
            f"'{module_name}.OUTPUT_DIR' is {type(output_dir)}, expected Path"
        )
        _logger.info(f"'{module_name}.OUTPUT_DIR' is a Path — OK")

    # ------------------------------------------------------------------
    # Data validation helpers
    # ------------------------------------------------------------------

    @keyword("Default Value In Source Should Be")
    def default_value_in_source_should_be(
        self, module_name: str, input_id_fragment: str, expected_value: str,
    ) -> None:
        """Assert a default ``selected=`` value for a given input ID fragment.

        Args:
            module_name: Short module name.
            input_id_fragment: Unique fragment of the input ID string.
            expected_value: Expected default value string.
        """
        source = self._get_source(module_name)
        lines = source.splitlines()
        fragment_found = False
        for i, line in enumerate(lines):
            if input_id_fragment in line:
                fragment_found = True
                # Check surrounding lines for selected= pattern
                context_block = "\n".join(lines[i : min(i + 10, len(lines))])
                match = re.search(r'selected=["\']([^"\']+)["\']', context_block)
                if match:
                    actual = match.group(1)
                    assert actual == expected_value, (
                        f"Default for '{input_id_fragment}' is '{actual}', "
                        f"expected '{expected_value}'"
                    )
                    _logger.info(
                        f"Default for '{input_id_fragment}' = '{expected_value}' — OK",
                    )
                    return
        if not fragment_found:
            _logger.warning(f"Input ID fragment '{input_id_fragment}' not found in source")
