# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label focal loss with optional per-class weight (alpha) and class-balanced weighting.
    Inputs:
      logits: (batch, C)
      targets: (batch, C) with 0/1
      alpha: tensor shape (C,) or scalar
      gamma: focusing parameter
      reduction: 'mean' or 'sum' or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits -> probs
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)  # p_t
        modulating_factor = (1 - p_t) ** self.gamma

        loss = modulating_factor * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_factor * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def effective_num_weights(y_train, beta=0.9999):
    """
    Compute class-balanced weights (as scalar per class) using Effective Number of Samples.
    y_train: numpy array shape (N, C) of binary labels
    Returns: weights (C,) normalized to sum C
    """
    N, C = y_train.shape
    pos = y_train.sum(axis=0).astype(np.float64)
    effective_num = 1.0 - np.power(beta, pos)
    weights = (1.0 - beta) / (effective_num + 1e-12)
    # normalize to C
    weights = weights / (weights.sum()) * C
    return weights.astype(np.float32)
