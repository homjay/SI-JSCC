import torch
import torch.nn as nn


class LayerPowerNormalization(nn.Module):
    """
    Layer-wise Power Normalization.
    Normalizes the input tensor to have unit average power per sample.
    Formula: x_norm = x / sqrt(mean(x^2) + eps)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, channel_first=True):
        if channel_first:
            dims = list(range(2, x.ndim))
        else:
            dims = list(range(1, x.ndim - 1))
        power = x.pow(2).mean(dim=dims, keepdim=True)
        x = x / torch.sqrt(power + self.eps)
        return x
