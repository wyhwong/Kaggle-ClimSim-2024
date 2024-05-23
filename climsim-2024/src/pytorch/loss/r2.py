import torch

import src.logger


local_logger = src.logger.get_logger(__name__)


def r2_score_multivariate(y_hat: torch.Tensor, y: torch.Tensor) -> torch.tensor:
    """
    Calculate the R2 score for multivariate targets.

    Args:
        predictions (torch.Tensor): The predicted targets.
        targets (torch.Tensor): The ground truth targets.

    Returns:
        torch.Tensor: The R2 score.
    """

    r2 = 1.0 - r2_residual_multivariate(y_hat, y)

    return r2


def r2_residual_multivariate(y_hat: torch.Tensor, y: torch.Tensor) -> torch.tensor:
    """
    Calculate the R2 loss.

    Args:
        y_hat (torch.Tensor): The predicted targets.
        y (torch.Tensor): The ground truth targets.

    Returns:
        torch.Tensor: The R2 loss.
    """

    mean_targets = torch.mean(y, dim=0)

    ss_total = torch.sum(torch.square(y - mean_targets), dim=0)
    ss_total[ss_total == 0] = 1.0
    ss_residual = torch.sum(torch.square(y - y_hat), dim=0)

    uni_r2_loss = ss_residual / ss_total
    loss = torch.nanmean(uni_r2_loss[torch.isfinite(uni_r2_loss)])

    local_logger.debug("R2 residual computed: %4f, n_dim: %s", loss, uni_r2_loss.shape)

    return loss
