import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.schemas.climsim import INPUT_COLUMNS, OUTPUT_COLUMNS


def compute_dataset_statistics(parquet_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the mean and variance of the dataset.

    Args:
        parquet_path (str): Path to the parquet file

    Returns:
        pd.DataFrame: Mean and Variance of the inputs
        pd.DataFrame: Mean and Variance of the outputs
    """

    parquet = pq.ParquetFile(parquet_path, memory_map=True, buffer_size=10)
    grp_x_mean, grp_x_var, grp_y_mean, grp_y_var, grp_nrows = [], [], [], [], []

    for row_group in range(parquet.num_row_groups):
        df_var = parquet.read_row_group(row_group).to_pandas()
        grp_x_mean.append(df_var[INPUT_COLUMNS].mean())
        grp_x_var.append(df_var[INPUT_COLUMNS].var())
        grp_y_mean.append(df_var[OUTPUT_COLUMNS].mean())
        grp_y_var.append(df_var[OUTPUT_COLUMNS].var())
        grp_nrows.append(df_var.shape[0])

    df_grp_x_mean = pd.DataFrame(grp_x_mean)
    df_grp_x_var = pd.DataFrame(grp_x_var)
    df_grp_y_mean = pd.DataFrame(grp_y_mean)
    df_grp_y_var = pd.DataFrame(grp_y_var)
    ds_nrows = pd.Series(grp_nrows)

    x_mean = df_grp_x_mean.apply(lambda x: (x * ds_nrows).sum() / ds_nrows.sum())
    y_mean = df_grp_y_mean.apply(lambda x: (x * ds_nrows).sum() / ds_nrows.sum())
    x_var = combine_var_from_groups(df_var=df_grp_x_var, df_mean=df_grp_x_mean, grp_nrows=ds_nrows)
    y_var = combine_var_from_groups(df_var=df_grp_y_var, df_mean=df_grp_y_mean, grp_nrows=ds_nrows)

    ds_min, ds_max = get_extremes(parquet)
    x_min, x_max = ds_min[INPUT_COLUMNS], ds_max[INPUT_COLUMNS]
    y_min, y_max = ds_min[OUTPUT_COLUMNS], ds_max[OUTPUT_COLUMNS]

    df_x_stats = pd.DataFrame({"mean": x_mean, "var": x_var, "min": x_min, "max": x_max})
    df_y_stats = pd.DataFrame({"mean": y_mean, "var": y_var, "min": y_min, "max": y_max})

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


def get_extremes(parquet: pq.ParquetFile) -> tuple[pd.Series, pd.Series]:
    """
    Get the maximum and minimum values of the dataset.

    Args:
        parquet (pq.ParquetFile): Parquet file

    Returns:
        pd.Series: Minimum values of the dataset
        pd.Series: Maximum values of the dataset
    """

    cols = INPUT_COLUMNS + OUTPUT_COLUMNS
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
