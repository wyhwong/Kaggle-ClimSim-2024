import torch

import src.logger


local_logger = src.logger.get_logger(__name__)


def r2_score_multivariate(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate the R2 score for multivariate targets.

    Args:
        predictions (torch.Tensor): The predicted targets.
        targets (torch.Tensor): The ground truth targets.

    Returns:
        torch.Tensor: The R2 score.
    """

    mean_targets = torch.mean(targets, dim=0)

    ss_total = torch.sum(targets - mean_targets, dim=0)
    ss_total[ss_total == 0] = 1.0
    ss_residual = torch.sum(targets - predictions, dim=0)

    uni_r2 = 1.0 - torch.square(ss_residual / ss_total)
    r2 = torch.nanmean(uni_r2[torch.isfinite(uni_r2)])

    local_logger.debug("R2 computed: %4f, n_dim: %s", r2, uni_r2.shape)

    return r2