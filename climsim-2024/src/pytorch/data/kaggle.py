import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

import src.env
import src.logger
import src.schemas.climsim


local_logger = src.logger.get_logger(__name__)


def output_compressed_parquet(
    model: nn.Module,
    df: pd.DataFrame,
    input_cols: list[str],
    weights: pd.DataFrame,
    output_dir: str = ".",
) -> None:
    """
    Output the submission file in the format of a gzipped parquet file.
    TODO: Replace pandas with polars for better performance.

    Args:
        model (nn.Module): The model to be used for inference
        df (pd.DataFrame): The input data
        weights (pd.DataFrame): The weights for each output column
        output_dir (str): The output directory

    Returns:
        None
    """

    inputs_set = df.loc[:, input_cols].values
    outputs_set = []

    model.eval()

    # Use tqdm for progress bar
    for i in tqdm(range(len(inputs_set)), desc="Processing Samples"):
        inputs = torch.Tensor(inputs_set[i : i + 1]).float().to(src.env.DEVICE)
        with torch.no_grad():
            output = model(inputs)
        outputs_set.append(output.cpu().numpy())

    df_output = pd.DataFrame(
        np.concatenate(outputs_set, axis=0),
        columns=src.schemas.climsim.OUTPUT_COLUMNS,
    )
    df_output["sample_id"] = df["sample_id"]

    for col in src.schemas.climsim.OUTPUT_COLUMNS:
        df_output[col] *= weights[col]

    df_output.to_parquet(
        path=f"{output_dir}/submission.parquet",
        index=False,
        engine="pyarrow",
    )
