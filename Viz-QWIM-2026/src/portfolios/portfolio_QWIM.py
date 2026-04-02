"""Portfolio Module.

This module provides functionality for managing financial portfolios with time-varying weights.

Classes
-------
portfolio_QWIM
    A class representing a financial portfolio with weights of components over time.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Self

import polars as pl


class portfolio_QWIM:
    """A class representing a financial portfolio with weights of components over time.

    The portfolio is represented as a DataFrame with dates and component weights.

    Attributes
    ----------
    m_portfolio_weights : pl.DataFrame
        DataFrame containing dates and portfolio weights.
    m_portfolio_components : List[str]
        List of strings containing names of the portfolio components.
    m_num_components : int
        Number of portfolio components.
    m_portfolio_name : str
        Name of the portfolio.

    Examples
    --------
    Creating a portfolio with equal weights:

    >>> import polars as pl
    >>> from src.portfolios.portfolio_QWIM import portfolio_QWIM
    >>> # Create portfolio with equal weights for three components
    >>> p1 = portfolio_QWIM(
    ...     name_portfolio="Tech Portfolio", names_components=["AAPL", "MSFT", "GOOG"]
    ... )
    >>> p1.get_portfolio_components
    ['AAPL', 'MSFT', 'GOOG']
    >>> p1.get_num_components
    3
    >>> p1.get_portfolio_name
    'Tech Portfolio'

    Creating a portfolio from existing data:

    >>> # Create portfolio from existing data
    >>> df = pl.DataFrame(
    ...     {"Date": ["2023-01-01", "2023-02-01"], "AAPL": [0.5, 0.6], "MSFT": [0.5, 0.4]}
    ... )
    >>> p2 = portfolio_QWIM(name_portfolio="Apple-Microsoft Mix", portfolio_weights=df)
    >>> p2.get_num_components
    2
    >>> p2.get_portfolio_name
    'Apple-Microsoft Mix'
    """

    def __init__(
        self,
        name_portfolio: str,
        portfolio_weights: pl.DataFrame | None = None,
        names_components: list[str] | None = None,
        date_portfolio: str | datetime | pl.Date | pl.Datetime | None = None,
    ) -> None:
        """Initialize a portfolio object with enhanced error handling."""
        try:
            # Validate portfolio name
            self.m_validate_type_and_value(
                name_portfolio,
                "name_portfolio",
                str,
                additional_check=lambda x: len(x.strip()) > 0,
                error_msg="Portfolio name must be a non-empty string",
            )
            self.m_portfolio_name = name_portfolio.strip()

            # Determine initialization mode
            has_weights = portfolio_weights is not None
            has_components = names_components is not None

            if not has_weights and not has_components:
                raise ValueError("Either portfolio_weights or names_components must be provided")

            # Initialize from weights DataFrame
            if has_weights:
                if portfolio_weights is None:
                    raise ValueError("portfolio_weights cannot be None when has_weights is True")
                try:
                    # Validate DataFrame
                    self.m_validate_dataframe(portfolio_weights, "portfolio_weights")

                    # Extract component names
                    self.m_portfolio_components = [
                        item_col for item_col in portfolio_weights.columns if item_col != "Date"
                    ]
                    self.m_num_components = len(self.m_portfolio_components)

                    # Validate and potentially correct weights
                    corrected_weights = self.m_validate_weights(
                        portfolio_weights,
                        self.m_portfolio_components,
                    )

                    # Ensure Date is first column
                    cols = ["Date"] + self.m_portfolio_components
                    self.m_portfolio_weights = corrected_weights.select(cols)

                except Exception as e:
                    if isinstance(e, ValueError | TypeError):
                        raise
                    raise ValueError(f"Error initializing from weights DataFrame: {e!s}") from e
            else:
                if names_components is None:
                    raise ValueError("names_components cannot be None when has_weights is False")
                try:
                    # Validate component names
                    self.m_validate_type_and_value(
                        names_components,
                        "names_components",
                        list,
                        additional_check=lambda x: len(x) > 0,
                        error_msg="names_components cannot be empty",
                    )

                    # Validate each component name
                    for idx, idx_name in enumerate(names_components):
                        self.m_validate_type_and_value(
                            idx_name,
                            f"Component name at index {idx}",
                            str,
                            additional_check=lambda x: len(x.strip()) > 0,
                            error_msg=f"Component name at index {idx} must be a non-empty string",
                        )

                    # Validate uniqueness
                    if len(set(names_components)) != len(names_components):
                        raise ValueError("Component names must be unique")

                    # Store clean component names
                    self.m_portfolio_components = [
                        item_name.strip() for item_name in names_components
                    ]
                    self.m_num_components = len(self.m_portfolio_components)

                    # Parse m_date
                    parsed_date = self.m_parse_date(date_portfolio)

                    # Create equal weights
                    equal_weight = 1.0 / self.m_num_components

                    # Create DataFrame
                    data_temp: dict[str, Any] = {"Date": [parsed_date]}
                    for idx_component in self.m_portfolio_components:
                        data_temp[idx_component] = [equal_weight]

                    self.m_portfolio_weights = pl.DataFrame(data_temp)

                except Exception as e:
                    if isinstance(e, ValueError | TypeError):
                        raise
                    raise ValueError(f"Error initializing from component names: {e!s}") from e

        except Exception as e:
            # Add more context to the error
            if isinstance(e, ValueError | TypeError):
                raise
            raise RuntimeError(f"Unexpected error initializing portfolio: {e!s}") from e

    def m_initialize_from_dataframe(self, portfolio_weights: pl.DataFrame) -> None:
        """Initialize portfolio from a DataFrame containing weights.

        Parameters
        ----------
        portfolio_weights : pl.DataFrame
            DataFrame with dates and portfolio weights.

        Raises
        ------
        ValueError
            If portfolio_weights does not have a 'Date' column.

        Warning
        -------
        This method assumes that all non-Date columns are portfolio components.
        Ensure your DataFrame doesn't contain any non-component columns besides 'Date'.
        """
        # Further validation (already checked for 'Date' and >1 column in __init__)
        if portfolio_weights.is_empty():
            raise ValueError("portfolio_weights DataFrame cannot be empty")

        # Extract component names (all columns except 'Date')
        components = [item_col for item_col in portfolio_weights.columns if item_col != "Date"]
        if not components:  # Should not happen due to __init__ check, but defensive
            raise ValueError(
                "portfolio_weights DataFrame must have component columns besides 'Date'",
            )

        # Validate component columns are numeric (or can be cast)
        df_clone = portfolio_weights.clone()  # Work on a clone for validation/casting
        for item_col in components:
            try:
                # Attempt to cast to Float64, allow nulls on failure initially
                df_clone = df_clone.with_columns(pl.col(item_col).cast(pl.Float64, strict=False))
                # Check if the column became all nulls after casting
                if df_clone.select(pl.col(item_col).is_null()).sum().item() == df_clone.height:
                    raise ValueError(
                        f"Component column '{item_col}' contains no valid numeric data.",
                    )
            except Exception as e:
                raise ValueError(
                    f"Error validating or casting component column '{item_col}': {e}",
                ) from e
        # Validate weights
        df_clone = self.m_validate_weights(df_clone, components)

        # Store the validated/potentially cast dataframe
        self.m_portfolio_weights = df_clone
        self.m_portfolio_components = components
        self.m_num_components = len(self.m_portfolio_components)

        # Validate that Date is the first column
        if portfolio_weights.columns[0] != "Date":
            # Reorder columns to ensure Date is first
            cols = ["Date"] + self.m_portfolio_components
            self.m_portfolio_weights = self.m_portfolio_weights.select(cols)

    def m_initialize_from_components(
        self,
        names_components: list[str],
        date_portfolio: str | datetime | None = None,
    ) -> None:
        r"""Initialize portfolio with equal weights for given components.

        Parameters
        ----------
        names_components : List[str]
            Names of portfolio components.
        date_portfolio : str or datetime, optional
            Date for the portfolio weights, defaults to current date if not provided.

        Notes
        -----
        Equal weights are calculated as:

        .. math::
           w_i = \\frac{1}{n}

        Where:

        - :math:`w_i` is the weight of component :math:`i`
        - :math:`n` is the number of components
        """
        # Set the date with validation
        formatted_date = self.m_parse_date(date_portfolio)

        # Calculate equal weights
        num_components = len(names_components)
        if num_components == 0:  # Should be caught by __init__, but defensive
            raise ValueError("Cannot initialize with zero components")
        equal_weight = 1.0 / num_components

        # Create the dataframe with one row
        data_date: dict[str, Any] = {"Date": [formatted_date]}  # Use validated date
        for idx_component in names_components:
            data_date[idx_component] = [equal_weight]

        # Store the portfolio data
        self.m_portfolio_weights = pl.DataFrame(data_date)
        self.m_portfolio_components = names_components
        self.m_num_components = num_components

    def m_validate_type_and_value(
        self,
        var: Any,
        var_name: str,
        expected_type: type | tuple[type, ...],
        allow_none: bool = False,
        additional_check: Callable[[Any], bool] | None = None,
        error_msg: str | None = None,
    ) -> None:
        """Validate variable type and optionally its value.

        Parameters
        ----------
        var : Any
            Variable to validate
        var_name : str
            Name of the variable for error messages
        expected_type : type or tuple of types
            Expected type(s) for the variable
        allow_none : bool, optional
            Whether None is allowed, by default False
        additional_check : callable, optional
            Additional validation function taking var as input, by default None
        error_msg : str, optional
            Custom error message, by default None

        Returns
        -------
        bool
            True if validation passes

        Raises
        ------
        TypeError
            If type validation fails
        ValueError
            If additional validation fails
        """
        # Check for None if not allowed
        if var is None:
            if allow_none:
                return
            raise ValueError(f"{var_name} cannot be None")

        # Type check
        if not isinstance(var, expected_type):
            type_names = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else " or ".join(t.__name__ for t in expected_type)
            )
            raise TypeError(f"{var_name} must be of type {type_names}, got {type(var).__name__}")

        # Additional validation
        if additional_check is not None and not additional_check(var):
            raise ValueError(error_msg or f"Invalid value for {var_name}")

    def m_validate_dataframe(
        self,
        df: Any,
        df_name: str = "input_DataFrame",
    ) -> bool:
        """Thoroughly validate a DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame to validate
        df_name : str, optional
            Name of DataFrame for error messages, by default "DataFrame"

        Returns
        -------
        bool
            True if validation passes

        Raises
        ------
        TypeError
            If df is not a pl.DataFrame
        ValueError
            If validation fails
        """
        # Type check
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"{df_name} must be a polars DataFrame, got {type(df).__name__}")

        # Check if empty
        if df.is_empty():
            raise ValueError(f"{df_name} cannot be empty")

        # Check for required Date column
        if "Date" not in df.columns:
            raise ValueError(f"{df_name} must have a 'Date' column")

        # Check number of columns
        if len(df.columns) < 2:
            raise ValueError(f"{df_name} must have at least one component column besides 'Date'")

        # Validate Date column
        try:
            # Check Date column type and try to cast if needed
            date_col_type = df.schema["Date"]

            if date_col_type not in (pl.Date, pl.Datetime, pl.Utf8):
                # Try to convert the column
                df = df.with_columns(pl.col("Date").cast(pl.Date, strict=False))

                # Check if conversion worked
                if df.select(pl.col("Date").is_null()).sum().item() == df.height:
                    raise ValueError(
                        f"Date column in {df_name} could not be converted to a valid date format",
                    )
        except Exception as e:
            raise ValueError(f"Error validating Date column in {df_name}: {e!s}") from e

        # Identify component columns
        component_cols = [item_col for item_col in df.columns if item_col != "Date"]

        # Validate component columns are numeric
        for item_col in component_cols:
            try:
                # Count nulls before casting
                nulls_before = df.select(pl.col(item_col).is_null()).sum().item()

                # Try to cast to numeric
                test_df = df.with_columns(pl.col(item_col).cast(pl.Float64, strict=False))

                # Count nulls after casting
                nulls_after = test_df.select(pl.col(item_col).is_null()).sum().item()

                # Check if casting introduced new nulls
                if nulls_after > nulls_before:
                    raise ValueError(
                        f"Column '{item_col}' contains non-numeric values that cannot be converted",
                    )

                # Check for all nulls
                if nulls_after == df.height:
                    raise ValueError(f"Column '{item_col}' contains no valid numeric data")

            except Exception as e:
                raise ValueError(f"Error validating component column '{item_col}': {e!s}") from e
        return True

    def m_validate_weights(
        self,
        df: pl.DataFrame,
        component_cols: list[str],
    ) -> pl.DataFrame:
        """Validate portfolio weights.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with weights
        component_cols : list
            List of component column names

        Returns
        -------
        pl.DataFrame
            Validated and potentially corrected DataFrame

        Raises
        ------
        ValueError
            If weights are invalid and cannot be corrected
        """
        try:
            # Create a clone to work with
            validated_df = df.clone()

            # Check for negative weights
            for item_col in component_cols:
                has_negative = validated_df.select(pl.col(item_col) < 0).to_series().any()
                if has_negative:
                    # Log warning and set negative weights to 0
                    print(f"Warning: Negative weights found in '{item_col}' and set to 0")
                    validated_df = validated_df.with_columns(
                        pl.when(pl.col(item_col) < 0)
                        .then(0)
                        .otherwise(pl.col(item_col))
                        .alias(item_col),
                    )

            # Check row sums
            row_sums = validated_df.select(
                [
                    pl.sum_horizontal(component_cols).alias("sum"),
                ],
            )

            # Add row index for better error reporting
            row_sums = row_sums.with_row_index()

            # Check for rows with sum == 0
            zero_sum_rows = row_sums.filter(pl.col("sum") == 0)
            if not zero_sum_rows.is_empty():
                row_indices = zero_sum_rows.select("index").to_series().to_list()
                dates = validated_df.select(pl.col("Date")).filter(
                    pl.int_range(0, validated_df.height).is_in(row_indices),
                )
                date_str = ", ".join(str(d) for d in dates.to_series().to_list())
                raise ValueError(f"Row(s) with all zero weights found for date(s): {date_str}")

            # Check for rows with sum far from 1.0
            tolerance = 0.001  # 0.1% tolerance
            invalid_sum_rows = row_sums.filter(
                (pl.col("sum") < 1 - tolerance) | (pl.col("sum") > 1 + tolerance),
            )

            if not invalid_sum_rows.is_empty():
                # Normalize weights for rows with sums not close to 1.0
                for idx in invalid_sum_rows.select("index").to_series().to_list():
                    row_sum = row_sums.filter(pl.col("index") == idx).select("sum")[0, 0]
                    if row_sum > 0:  # Only normalize if sum is positive
                        for item_col in component_cols:
                            validated_df = validated_df.with_columns(
                                [
                                    pl.when(pl.int_range(0, validated_df.height) == idx)
                                    .then(pl.col(item_col) / row_sum)
                                    .otherwise(pl.col(item_col))
                                    .alias(item_col),
                                ],
                            )
                        print(
                            f"Warning: Weights in row {idx} summed to {row_sum}, normalized to 1.0",
                        )

            return validated_df

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error validating weights: {e!s}") from e

    def m_parse_date(
        self,
        date_input: str | datetime | pl.Date | pl.Datetime | None,
    ) -> str:
        """Parse various date formats into a standardized string format.

        Parameters
        ----------
        date_input : Union[str, datetime, pl.Date, pl.Datetime, None]
            Date in various possible formats

        Returns
        -------
        str
            Date in YYYY-MM-DD format

        Raises
        ------
        ValueError
            If date cannot be parsed
        TypeError
            If date is of unsupported type
        """
        try:
            if date_input is None:
                # Use current date
                return datetime.now().strftime("%Y-%m-%d")

            if isinstance(date_input, str):
                # Try to parse string date in various formats
                for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"):
                    try:
                        parsed_date = datetime.strptime(date_input, fmt)
                        return parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
                # If we get here, none of the formats worked
                raise ValueError(
                    f"Could not parse date string '{date_input}'. Use YYYY-MM-DD format.",
                )

            if isinstance(date_input, datetime):
                return date_input.strftime("%Y-%m-%d")

            if isinstance(date_input, pl.Date | pl.Datetime):
                # Create a temporary DataFrame to format the date
                temp_df = pl.DataFrame({"temp_date": [date_input]})
                try:
                    formatted_date = temp_df.select(pl.col("temp_date").dt.strftime("%Y-%m-%d"))[
                        0,
                        0,
                    ]
                    if formatted_date is None:
                        raise ValueError("Date conversion resulted in None")
                    return formatted_date
                except Exception as e:
                    raise ValueError(f"Could not format Polars date: {e!s}") from e

            # If we get here, the type is unsupported
            raise TypeError(f"Unsupported date type: {type(date_input).__name__}")

        except Exception as e:
            if isinstance(e, ValueError | TypeError):
                raise
            raise ValueError(f"Error parsing date: {e!s}") from e

    def get_portfolio_weights(self) -> pl.DataFrame:
        """Get the portfolio weights DataFrame.

        Returns
        -------
        pl.DataFrame
            A copy of the portfolio weights DataFrame.

        Notes
        -----
        Returns a clone of the internal DataFrame to prevent unintended modifications.
        """
        return self.m_portfolio_weights.clone()

    @property
    def get_portfolio_components(self) -> list[str]:
        """Get the list of portfolio components.

        Returns
        -------
        List[str]
            A copy of the list of portfolio component names.
        """
        return self.m_portfolio_components.copy()

    @property
    def get_num_components(self) -> int:
        """Get the number of portfolio components.

        Returns
        -------
        int
            The number of components in the portfolio.
        """
        return self.m_num_components

    @property
    def get_portfolio_name(self) -> str:
        """Get the name of the portfolio.

        Returns
        -------
        str
            The name of the portfolio.

        See Also
        --------
        set_portfolio_name : Method to set a new portfolio name
        """
        return self.m_portfolio_name

    def set_portfolio_name(self, name: str) -> None:
        """Set a new name for the portfolio.

        Parameters
        ----------
        name : str
            The new portfolio name.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Portfolio name must be a non-empty string")
        self.m_portfolio_name = name

    def __str__(self) -> str:
        """Return string representation of the portfolio.

        Returns
        -------
        str
            A string representation showing the portfolio name, number of components,
            and the portfolio weights.
        """
        return (
            f"Portfolio '{self.m_portfolio_name}' with {self.m_num_components} components:"
            f"\n{self.m_portfolio_weights}"
        )

    def __repr__(self) -> str:
        """Return string representation for developers.

        Returns
        -------
        str
            A detailed string representation showing the portfolio name, components,
            and number of dates.
        """
        return (
            f"portfolio_QWIM(name='{self.m_portfolio_name}', "
            f"components={self.m_portfolio_components}, "
            f"dates={len(self.m_portfolio_weights)} rows)"
        )

    def add_weights(
        self,
        input_date: str | datetime | pl.Date | pl.Datetime,
        weights: dict[str, float] | None = None,
        validate: bool = True,
    ) -> Self:
        """Add a new row of weights for a specific date.

        Parameters
        ----------
        input_date : Union[str, datetime, pl.Date, pl.Datetime]
            Date for the new weights
        weights : dict, optional
            Dictionary mapping component names to weights
            If None, uses equal weights for all components
        validate : bool, optional
            Whether to validate and normalize weights, by default True

        Returns
        -------
        portfolio_QWIM
            Self for method chaining

        Raises
        ------
        ValueError
            If date or weights are invalid
        """
        try:
            # Parse date
            parsed_date = self.m_parse_date(input_date)

            # Check if date already exists
            date_exists = self.m_portfolio_weights.filter(pl.col("Date") == parsed_date).height > 0
            if date_exists:
                raise ValueError(f"Weights for date {parsed_date} already exist")

            # Create weights dictionary if not provided
            if weights is None:
                # Use equal weights
                equal_weight = 1.0 / self.m_num_components
                weights = dict.fromkeys(self.m_portfolio_components, equal_weight)
            else:
                # Validate weights dictionary
                self.m_validate_type_and_value(
                    weights,
                    "weights",
                    dict,
                    error_msg="weights must be a dictionary mapping component names to weight values",
                )

                # Check for missing components
                missing_components = [
                    item_comp
                    for item_comp in self.m_portfolio_components
                    if item_comp not in weights
                ]
                if missing_components:
                    raise ValueError(
                        f"Missing weights for components: {', '.join(missing_components)}",
                    )

                # Check for extra components
                extra_components = [
                    item_comp
                    for item_comp in weights
                    if item_comp not in self.m_portfolio_components
                ]
                if extra_components:
                    raise ValueError(
                        f"Unexpected components in weights: {', '.join(extra_components)}",
                    )

                # Check for negative weights
                neg_components = [item_comp for item_comp, idx_w in weights.items() if idx_w < 0]
                if neg_components:
                    if validate:
                        # Zero out negative weights
                        for idx_comp in neg_components:
                            print(f"Warning: Negative weight for '{idx_comp}' set to 0")
                            weights[idx_comp] = 0
                    else:
                        raise ValueError(
                            f"Negative weights not allowed for: {', '.join(neg_components)}",
                        )

                # Check sum of weights
                weight_sum = sum(weights.values())
                if abs(weight_sum - 1.0) > 0.001:  # More than 0.1% off
                    if validate and weight_sum > 0:
                        # Normalize weights
                        weights = {
                            item_comp: idx_w / weight_sum for item_comp, idx_w in weights.items()
                        }
                        print(f"Warning: Weights summed to {weight_sum}, normalized to 1.0")
                    else:
                        raise ValueError(f"Sum of weights ({weight_sum}) is not close to 1.0")

            # Create new row
            new_row: dict[str, Any] = {"Date": parsed_date}
            new_row.update(weights)

            # Append to DataFrame
            new_df = pl.DataFrame([new_row])
            self.m_portfolio_weights = pl.concat([self.m_portfolio_weights, new_df], how="vertical")

            # Sort by date
            self.m_portfolio_weights = self.m_portfolio_weights.sort("Date")

            return self

        except Exception as e:
            if isinstance(e, ValueError | TypeError):
                raise
            raise ValueError(f"Error adding weights: {e!s}") from e

    def modify_weights(
        self,
        input_date: str | datetime | pl.Date | pl.Datetime,
        new_weights: dict[str, float],
        validate: bool = True,
    ) -> Self:
        """Modify weights for a specific date.

        Parameters
        ----------
        input_date : Union[str, datetime, pl.Date, pl.Datetime]
            Date of weights to modify
        new_weights : dict
            Dictionary mapping component names to new weights
        validate : bool, optional
            Whether to validate and normalize weights, by default True

        Returns
        -------
        portfolio_QWIM
            Self for method chaining

        Raises
        ------
        ValueError
            If date or weights are invalid
        """
        try:
            # Parse date
            parsed_date = self.m_parse_date(input_date)

            # Check if date exists
            date_filter = self.m_portfolio_weights.filter(pl.col("Date") == parsed_date)
            if date_filter.is_empty():
                raise ValueError(f"No weights exist for date {parsed_date}")

            # Validate weights dictionary
            self.m_validate_type_and_value(
                new_weights,
                "new_weights",
                dict,
                error_msg="new_weights must be a dictionary mapping component names to weight values",
            )

            # Check for invalid components
            invalid_components = [
                item_comp
                for item_comp in new_weights
                if item_comp not in self.m_portfolio_components
            ]
            if invalid_components:
                raise ValueError(
                    f"Invalid components in new_weights: {', '.join(invalid_components)}",
                )

            # Get current weights
            current_weights = {}
            for item_comp in self.m_portfolio_components:
                current_weights[item_comp] = date_filter.select(pl.col(item_comp))[0, 0]

            # Update with new weights
            current_weights.update(new_weights)

            # Check for negative weights
            neg_components = [
                item_comp for item_comp, idx_w in current_weights.items() if idx_w < 0
            ]
            if neg_components:
                if validate:
                    # Zero out negative weights
                    for item_comp in neg_components:
                        print(f"Warning: Negative weight for '{item_comp}' set to 0")
                        current_weights[item_comp] = 0
                else:
                    raise ValueError(
                        f"Negative weights not allowed for: {', '.join(neg_components)}",
                    )

            # Check sum of weights
            weight_sum = sum(current_weights.values())
            if abs(weight_sum - 1.0) > 0.001:  # More than 0.1% off
                if validate and weight_sum > 0:
                    # Normalize weights
                    current_weights = {
                        item_comp: idx_w / weight_sum
                        for item_comp, idx_w in current_weights.items()
                    }
                    print(f"Warning: Modified weights summed to {weight_sum}, normalized to 1.0")
                else:
                    raise ValueError(f"Sum of modified weights ({weight_sum}) is not close to 1.0")

            # Update DataFrame
            for item_comp, item_weight in current_weights.items():
                self.m_portfolio_weights = self.m_portfolio_weights.with_columns(
                    [
                        pl.when(pl.col("Date") == parsed_date)
                        .then(pl.lit(item_weight))
                        .otherwise(pl.col(item_comp))
                        .alias(item_comp),
                    ],
                )

            return self

        except Exception as e:
            if isinstance(e, ValueError | TypeError):
                raise
            raise ValueError(f"Error modifying weights: {e!s}") from e

    def validate_all_weights(
        self,
        normalize: bool = True,
    ) -> Self:
        """Validate all weights in the portfolio.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize rows where sum != 1.0, by default True

        Returns
        -------
        portfolio_QWIM
            Self for method chaining

        Notes
        -----
        This method checks for and optionally corrects:
        - Negative weights (set to 0)
        - Rows where weights do not sum to 1.0 (normalize if normalize=True)
        """
        try:
            component_cols = self.m_portfolio_components

            # Validate and potentially correct weights
            corrected_weights = self.m_validate_weights(self.m_portfolio_weights, component_cols)

            # Ensure Date is first column
            cols = ["Date"] + component_cols
            self.m_portfolio_weights = corrected_weights.select(cols)

            return self

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error validating weights: {e!s}") from e
