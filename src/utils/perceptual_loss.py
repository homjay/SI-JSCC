#!/usr/bin/env python3

import lpips


def lpips_loss_fn():
    """
    Initialize and return the LPIPS loss function.
    """
    # Use the default LPIPS model
    loss_fn = lpips.LPIPS(net="squeeze")
    for param in loss_fn.parameters():
        param.requires_grad = False
    return loss_fn
