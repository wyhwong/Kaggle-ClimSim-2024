from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.schemas.climsim import INPUT_COLUMNS, OUTPUT_COLUMNS


def compute_dataset_statistics(
    parquet_path: str,
    input_cols: Optional[list[str]] = None,
    output_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the mean and variance of the dataset.

    Args:
        parquet_path (str): Path to the parquet file
        input_cols (Optional[list[str]]): Input columns
        output_cols (Optional[list[str]]): Output columns

    Returns:
        pd.DataFrame: Mean and Variance of the inputs
        pd.DataFrame: Mean and Variance of the outputs
    """

    parquet = pq.ParquetFile(parquet_path, memory_map=True, buffer_size=10)

    input_cols = input_cols or INPUT_COLUMNS
    output_cols = output_cols or OUTPUT_COLUMNS

    # Get the maximum and minimum values of the dataset
    ds_min, ds_max = get_extremes(parquet, input_cols, output_cols)
    x_min, x_max = ds_min[input_cols], ds_max[input_cols]
    y_min, y_max = ds_min[output_cols], ds_max[output_cols]

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_scaling, y_scaling = x_range.copy(), y_range.copy()
    x_scaling[x_scaling == 0] = 1.0
    y_scaling[y_scaling == 0] = 1.0

    # Compute the mean and variance of the dataset
    grp_x_mean, grp_x_var, grp_y_mean, grp_y_var, grp_nrows = [], [], [], [], []
    grp_norm_x_mean, grp_norm_x_var, grp_norm_y_mean, grp_norm_y_var = [], [], [], []
    for row_group in range(parquet.num_row_groups):
        df_grp = parquet.read_row_group(row_group).to_pandas()
        grp_x_mean.append(df_grp[input_cols].mean())
        grp_x_var.append(df_grp[input_cols].var())
        grp_y_mean.append(df_grp[output_cols].mean())
        grp_y_var.append(df_grp[output_cols].var())
        grp_nrows.append(df_grp.shape[0])

        grp_norm_x = (df_grp[input_cols] - x_min) / x_scaling
        grp_norm_y = (df_grp[output_cols] - y_min) / y_scaling
        grp_norm_x_mean.append(grp_norm_x.mean())
        grp_norm_x_var.append(grp_norm_x.var())
        grp_norm_y_mean.append(grp_norm_y.mean())
        grp_norm_y_var.append(grp_norm_y.var())

    df_grp_x_mean = pd.DataFrame(grp_x_mean)
    df_grp_x_var = pd.DataFrame(grp_x_var)
    df_grp_y_mean = pd.DataFrame(grp_y_mean)
    df_grp_y_var = pd.DataFrame(grp_y_var)
    ds_nrows = pd.Series(grp_nrows)

    df_grp_norm_x_mean = pd.DataFrame(grp_norm_x_mean)
    df_grp_norm_x_var = pd.DataFrame(grp_norm_x_var)
    df_grp_norm_y_mean = pd.DataFrame(grp_norm_y_mean)
    df_grp_norm_y_var = pd.DataFrame(grp_norm_y_var)

    x_mean = df_grp_x_mean.apply(lambda x: (x * ds_nrows).sum() / ds_nrows.sum())
    y_mean = df_grp_y_mean.apply(lambda x: (x * ds_nrows).sum() / ds_nrows.sum())
    x_var = combine_var_from_groups(df_var=df_grp_x_var, df_mean=df_grp_x_mean, grp_nrows=ds_nrows)
    y_var = combine_var_from_groups(df_var=df_grp_y_var, df_mean=df_grp_y_mean, grp_nrows=ds_nrows)

    norm_x_mean = df_grp_norm_x_mean.apply(lambda x: (x * ds_nrows).sum() / ds_nrows.sum())
    norm_y_mean = df_grp_norm_y_mean.apply(lambda x: (x * ds_nrows).sum() / ds_nrows.sum())
    norm_x_var = combine_var_from_groups(df_var=df_grp_norm_x_var, df_mean=df_grp_norm_x_mean, grp_nrows=ds_nrows)
    norm_y_var = combine_var_from_groups(df_var=df_grp_norm_y_var, df_mean=df_grp_norm_y_mean, grp_nrows=ds_nrows)

    df_x_stats = pd.DataFrame(
        {
            "min": x_min,
            "max": x_max,
            "range": x_range,
            "mean": x_mean,
            "var": x_var,
            "std": np.sqrt(x_var),
            "norm_mean": norm_x_mean,
            "norm_var": norm_x_var,
            "norm_std": np.sqrt(norm_x_var),
        }
    ).T
    df_y_stats = pd.DataFrame(
        {
            "min": y_min,
            "max": y_max,
            "range": y_range,
            "mean": y_mean,
            "var": y_var,
            "std": np.sqrt(y_var),
            "norm_mean": norm_y_mean,
            "norm_var": norm_y_var,
            "norm_std": np.sqrt(norm_y_var),
        }
    ).T

    return (df_x_stats, df_y_stats)


def combine_var_from_groups(df_var: pd.DataFrame, df_mean: pd.DataFrame, grp_nrows: pd.Series) -> pd.Series:
    """
    Combine the variance of different groups.
    Reference: https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
    mean = (n1 * mean1 + n2 * mean2) / (n1 + n2)
    variance = ( ( n1 - 1 ) * var1 + ( n2 - 1 ) * var2 ) / ( n1 + n2 - 1 )
               + (n1*n2) * (mean1 - mean2)^2 / (n1 + n2) / (n1 + n2 - 1)

    Args:
        df_var (pd.DataFrame): Variance of each group
        df_mean (pd.DataFrame): Mean of each group
        grp_nrows (pd.Series): Number of rows in each group

    Returns:
        pd.Series: Combined variance
    """

    var = df_var.iloc[0]
    mean = df_mean.iloc[0]
    n = grp_nrows[0]

    for i, grp_var in df_var.iterrows():
        if i == 0:
            continue

        grp_n, grp_mean = grp_nrows[i], df_mean.iloc[i]
        term_1 = ((n - 1) * var + (grp_n - 1) * grp_var) / (n + grp_n - 1)
        term_2 = (n * grp_n) * (mean - grp_mean) ** 2 / (n + grp_n) / (n + grp_n - 1)
        var = term_1 + term_2
        n += grp_n

    return var


def get_extremes(
    parquet: pq.ParquetFile,
    input_cols: list[str],
    output_cols: list[str],
) -> tuple[pd.Series, pd.Series]:
    """
    Get the maximum and minimum values of the dataset.

    Args:
        parquet (pq.ParquetFile): Parquet file
        input_cols (list[str]): Input columns
        output_cols (list[str]): Output columns

    Returns:
        pd.Series: Minimum values of the dataset
        pd.Series: Maximum values of the dataset
    """

    cols = input_cols + output_cols
    max_values = {col: -np.inf for col in cols}
    min_values = {col: np.inf for col in cols}

    # Go through metadata of each row group to get the max and min values
    for row_group in range(parquet.num_row_groups):
        meta = parquet.metadata.row_group(row_group)
        for idx in range(meta.num_columns):
            col_meta = meta.column(idx)
            col = col_meta.path_in_schema
            if col in cols:
                max_values[col] = max(max_values[col], col_meta.statistics.max)
                min_values[col] = min(min_values[col], col_meta.statistics.min)

    ds_min, ds_max = pd.Series(min_values), pd.Series(max_values)
    return ds_min, ds_max
