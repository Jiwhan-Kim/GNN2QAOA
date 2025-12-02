import torch
import torch.nn as nn


class MultiTargetMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, targets):
        # compute squared error to each target
        errs = (pred - targets)**2  # shape (3, 2 * n_layers)
        norms = torch.norm(errs, dim=1)

        # tau: temperature
        # larger tau: smoother, gradient spreads more evenly
        tau = 0.1
        min = -tau * torch.logsumexp(-norms / tau, dim=0)
        return min
