import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

import src.logger
import src.pytorch.dataset.base
import src.schemas.climsim


local_logger = src.logger.get_logger(__name__)


def _get_underperforming_mask(y_hat: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """Get a mask for underperforming targets.

    Args:
        y_hat (torch.Tensor): The predicted values
        y (torch.Tensor): The target values

    Returns:
        mask (np.ndarray): The mask for underperforming targets
    """

    mean_targets = torch.mean(y, dim=0)

    ss_total = torch.sum(torch.square(y - mean_targets), dim=0)
    ss_residual = torch.sum(torch.square(y - y_hat), dim=0)

    uni_r2 = 1.0 - ss_residual / ss_total
    uni_r2 = uni_r2.clip(-0.1, 1.0)

    mask = (uni_r2 < 0.0).cpu().numpy()
    local_logger.warning("Number of underperforming targets: %s", mask.sum())

    return mask


def output_submission_parquet(
    model: nn.Module,
    dataset: src.pytorch.dataset.base.DatasetBase,
    df: pd.DataFrame,
    weights: pd.DataFrame,
    calibration_batch_size: int = 16348,
    output_dir: str = ".",
) -> pd.DataFrame:
    """Output the submission file in the format of a gzipped parquet file.
    NOTE: The goal of calibration is to replace the underperforming targets with the mean.
          So that the model does not make predictions that are worse than the mean.

    Args:
        model (nn.Module): The model to be used for inference
        dataset (src.pytorch.dataset.base.DatasetBase): The dataset object (used for training)
        df (pd.DataFrame): The input data (features)
        weights (pd.DataFrame): The weights for each output column
        calibration_batch_size (int): The batch size for calibration
        output_dir (str): The output directory

    Returns:
        pd.DataFrame: The output data
    """

    # Set the model to evaluation mode
    model.eval()

    # Calibration: Find which outputs are underperforming
    x, y = dataset.get_batch(size=calibration_batch_size)
    y_hat = model(x)
    mask = _get_underperforming_mask(y_hat=y_hat, y=y)

    # Inference for submission
    inputs_set = dataset.preprocess_features(x=df.loc[:, dataset.input_cols].values)
    outputs_set: list[np.ndarray] = []
    for i in tqdm(range(len(inputs_set)), desc="Processing Samples"):
        inputs = inputs_set[i : i + 1]
        with torch.no_grad():
            output = model(inputs)
        outputs_set.append(output.cpu().numpy())

    df_output = pd.DataFrame(
        np.concatenate(outputs_set, axis=0),
        columns=dataset.output_cols,
    )

    # Replace underperforming targets with the mean
    outputs = dataset.postprocess_targets(df_output[dataset.output_cols].values)
    outputs[:, mask == 1] = dataset.y_stats.loc["mean"].values[mask == 1]
    df_output[dataset.output_cols] = outputs
    df_output = pd.concat([df[["sample_id"]], df_output], axis=1)

    # Apply weights
    for col in dataset.output_cols:
        df_output[col] *= weights[col]

    # Replace some columns with static values (according to the competition)
    for idx in range(12, 30):
        df_output[f"ptend_q0002_{idx}"] = -df[f"state_q0002_{idx}"].to_numpy() / 1200.0

    # Save the submission file
    # NOTE: Here we do not use compression=gzip because Kaggle failed to read the file
    #       Also, gzip compression does not save much space because parquet format is already efficient
    df_output.to_parquet(
        path=f"{output_dir}/submission.parquet",
        index=False,
        engine="pyarrow",
    )

    return df_output
