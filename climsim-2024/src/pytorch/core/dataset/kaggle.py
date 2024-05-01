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
    weights: pd.DataFrame,
    output_dir: str = ".",
) -> None:
    """
    Output the submission file in the format of a gzipped parquet file.
    TODO: Replace pandas with polars for better performance.

    Args:
        model: The PyTorch model to use for prediction.
        df: The input DataFrame.
        weights: The weights to apply to the output columns.
        output_dir: The path to the output file.

    Returns:
        None
    """

    inputs_set = df.loc[:, src.schemas.climsim.INPUT_COLUMNS].values
    outputs_set = []

    # Loop over the inputs
    # This loop is to avoid caching the whole dataset in GPU memory
    for inputs in tqdm(inputs_set):
        inputs = torch.tensor(inputs).float().to(src.env.DEVICE)
        output = model(inputs)
        outputs_set.append(output.cpu().detach().numpy())

    df_output = pd.DataFrame(
        outputs_set,
        columns=src.schemas.climsim.OUTPUT_COLUMNS,
    )
    df_output["sample_id"] = df["sample_id"]

    for col in src.schemas.climsim.OUTPUT_COLUMNS:
        df_output[col] *= weights[col]

    df_output.to_parquet(
        path=f"{output_dir}/submission.parquet.gz",
        index=False,
        compression="gzip",
        engine="pyarrow",
    )
