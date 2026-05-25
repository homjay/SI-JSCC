#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
SNR generation utilities for per-channel SNR training.
"""

import torch


def generate_per_channel_snr(
    num_channels: int,
    min_snr: float,
    max_snr: float,
    strategy: str = "random",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate per-channel SNR values.

    Args:
        num_channels: Number of channels
        min_snr: Minimum SNR in dB
        max_snr: Maximum SNR in dB
        strategy: Generation strategy, one of:
            - 'random': Random SNR for each channel
            - 'linear': Linearly increasing SNR from min to max
            - 'exponential': Exponentially increasing SNR
            - 'decreasing': Linearly decreasing SNR from max to min
        device: Target device for the tensor

    Returns:
        Tensor of shape (num_channels,) with SNR values in dB
    """
    if device is None:
        device = torch.device("cpu")

    if strategy == "random":
        # Random SNR for each channel
        snr = torch.rand(num_channels, device=device) * (max_snr - min_snr) + min_snr

    elif strategy == "linear":
        # Linearly increasing SNR
        snr = torch.linspace(min_snr, max_snr, num_channels, device=device)

    elif strategy == "exponential":
        # Exponentially increasing SNR
        # Convert to linear scale, interpolate, then back to dB
        min_linear = 10 ** (min_snr / 10)
        max_linear = 10 ** (max_snr / 10)
        linear_values = torch.linspace(
            min_linear, max_linear, num_channels, device=device
        )
        snr = 10 * torch.log10(linear_values)

    elif strategy == "decreasing":
        # Linearly decreasing SNR
        snr = torch.linspace(max_snr, min_snr, num_channels, device=device)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: 'random', 'linear', 'exponential', 'decreasing'"
        )

    return snr


def generate_batch_per_channel_snr(
    batch_size: int,
    num_channels: int,
    min_snr: float,
    max_snr: float,
    strategy: str = "random",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate per-channel SNR values for a batch.

    Args:
        batch_size: Batch size
        num_channels: Number of channels
        min_snr: Minimum SNR in dB
        max_snr: Maximum SNR in dB
        strategy: Generation strategy (see generate_per_channel_snr)
        device: Target device for the tensor

    Returns:
        Tensor of shape (batch_size, num_channels) with SNR values in dB

    Note:
        For 'random' strategy, each sample in the batch gets different random SNR values.
        For other strategies, all samples in the batch share the same SNR pattern.
    """
    if device is None:
        device = torch.device("cpu")

    if strategy == "random":
        # Each sample gets different random SNR
        snr = (
            torch.rand(batch_size, num_channels, device=device) * (max_snr - min_snr)
            + min_snr
        )
    else:
        # All samples share the same SNR pattern
        snr = generate_per_channel_snr(num_channels, min_snr, max_snr, strategy, device)
        snr = snr.unsqueeze(0).expand(batch_size, -1)

    return snr
