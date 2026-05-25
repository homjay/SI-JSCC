#!/usr/bin/env python3


import torch


class L1CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
