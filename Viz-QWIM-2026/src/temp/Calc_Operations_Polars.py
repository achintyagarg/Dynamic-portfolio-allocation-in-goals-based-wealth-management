import time
import polars as pl
import numpy as np


def fast_weighted_sum_GPT_4o(
        df: pl.DataFrame, 
        value_cols: np.ndarray, 
        weights: np.ndarray) -> pl.Series:
    """
    Calculate a weighted sum across rows for a Polars DataFrame using vectors/arrays.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing the values.
    value_cols : np.ndarray
        A NumPy array of column names containing the values to be weighted.
    weights : np.ndarray
        A NumPy array of weights corresponding to the value columns.

    Returns
    -------
    pl.Series
        A Polars Series containing the weighted sum for each row.
    """
    if len(value_cols) != len(weights):
        raise ValueError("The number of value columns must match the number of weights.")

    # Create a weighted sum expression using Polars expressions
    weighted_sum_expr = sum(pl.col(value_col) * weight for value_col, weight in zip(value_cols, weights))

    # Compute the weighted sum and return the result as a Series
    return df.select(weighted_sum_expr.alias("weighted_sum"))["weighted_sum"]


def fast_weighted_sum_Claude_3_7(df: pl.DataFrame, value_cols: np.ndarray, weights: np.ndarray) -> pl.Series:
    """
    Calculate a weighted sum across rows for a Polars DataFrame with maximum efficiency.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing the values.
    value_cols : np.ndarray
        A NumPy array of column names containing the values to be weighted.
    weights : np.ndarray
        A NumPy array of weights corresponding to the value columns.

    Returns
    -------
    pl.Series
        A Polars Series containing the weighted sum for each row.
    """
    if len(value_cols) != len(weights):
        raise ValueError("The number of value columns must match the number of weights.")

    # Create weighted expressions in one go using a list comprehension
    # instead of using a generator with sum() which creates intermediate objects
    products = [pl.col(col) * weight for col, weight in zip(value_cols, weights)]
    
    # Use sum_horizontal which is optimized for adding multiple columns
    # This is faster than Python's sum() function which adds iteratively
    return df.select(pl.sum_horizontal(products).alias("weighted_sum"))["weighted_sum"]


def polars_cumprod(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate cumulative product of columns in a Polars DataFrame with maximum efficiency.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    
    Returns
    -------
    pl.DataFrame
        DataFrame of the same shape as input, containing cumulative products of each column
    """
    # Apply cumulative product to all columns in a single operation
    # This avoids creating a Python list and iterating through column names
    # and processes the entire expression in Polars' execution engine
    exprs = [pl.col(col_name).cumprod().alias(col_name) for col_name in df.columns]
    return df.select(exprs)


# Define dimensions
num_cols_example = 30
num_rows_example = 230

# Generate column names in format "A_digit"
column_names = np.array([f"A_{idx+1}" for idx in range(num_cols_example)])

# Generate random weights between 0 and 1, then normalize to sum to 1.0
raw_weights = np.random.random(num_cols_example)
values_weights = raw_weights / raw_weights.sum()

# Verify the sum is 1.0
print(f"Sum of weights: {values_weights.sum()}")

# Generate random data between -1 and 1 for the dataframe
data_dict = {}
for col_name in column_names:
    data_dict[col_name] = np.random.uniform(-1, 1, num_rows_example)

# Create the Polars DataFrame
example_df = pl.DataFrame(data_dict)

# Store column names in values_cols
values_cols = column_names

print(f"DataFrame shape: {example_df.shape}")
print(f"Column names array length: {len(values_cols)}")
print(f"Weights array length: {len(values_weights)}")

# Measure the time for 1000 runs
num_runs = 1000

start_time_GPT_4o = time.time()

for _ in range(num_runs):
    result_GPT_4o = fast_weighted_sum_GPT_4o(example_df, values_cols, values_weights)

end_time_GPT_4o = time.time()

# Calculate the average time per run
average_time_GPT_4o = (end_time_GPT_4o - start_time_GPT_4o) / num_runs
print(f"Average computational time code_GPT_4o over {num_runs} runs: {average_time_GPT_4o:.6f} seconds")

start_time_Claude_3_7 = time.time()

for _ in range(num_runs):
    result_Claude_3_7 = fast_weighted_sum_Claude_3_7(example_df, values_cols, values_weights)

end_time_Claude_3_7 = time.time()

# Calculate the average time per run
average_time_Claude_3_7 = (end_time_Claude_3_7 - start_time_Claude_3_7) / num_runs
print(f"Average computational time code_Claude_3_7 over {num_runs} runs: {average_time_Claude_3_7:.6f} seconds")


