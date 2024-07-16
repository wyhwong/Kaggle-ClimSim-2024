import torch

import src.logger


local_logger = src.logger.get_logger(__name__)


def r2_score_multivariate(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate the R2 score for multivariate targets.

    Args:
        y_hat (torch.Tensor): The predicted values.
        y (torch.Tensor): The target values.

    Returns:
        r2 (torch.Tensor): The R2 score.
    """

    mean_targets = torch.mean(y, dim=0)

    ss_total = torch.sum(torch.square(y - mean_targets), dim=0)
    ss_residual = torch.sum(torch.square(y - y_hat), dim=0)

    uni_r2 = 1.0 - ss_residual / ss_total
    uni_r2 = uni_r2.clip(0.0, 1.0)

    r2 = torch.nanmean(uni_r2[torch.isfinite(uni_r2)])
    local_logger.debug("R2 computed: %4f, n_dim: %s", r2, uni_r2.shape)

    return r2
