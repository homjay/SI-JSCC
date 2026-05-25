#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Channel functions used by the JSCC autoencoder."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch


ChannelFn = Callable[[torch.Tensor, object], torch.Tensor]


def identity_channel(z: torch.Tensor, snr=None) -> torch.Tensor:
    return z


def _as_snr_tensor(snr, device: torch.device) -> Optional[torch.Tensor]:
    if snr is None:
        return None
    if torch.is_tensor(snr):
        return snr.to(device=device)
    return torch.tensor(snr, dtype=torch.float32, device=device)


def _signal_power(z: torch.Tensor, snr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if snr.ndim == 0 or snr.numel() == 1:
        dims = tuple(range(1, z.ndim))
        return snr.reshape(1, *([1] * (z.ndim - 1))), torch.mean(
            torch.abs(z) ** 2, dim=dims, keepdim=True
        )

    if snr.ndim == 1 and snr.shape[0] == z.shape[0]:
        dims = tuple(range(1, z.ndim))
        return snr.reshape(-1, *([1] * (z.ndim - 1))), torch.mean(
            torch.abs(z) ** 2, dim=dims, keepdim=True
        )

    # Per-channel SNR. The JSCC signal is flattened after real-to-complex
    # conversion, so a channel SNR vector may need to be repeated over spatial
    # positions.
    if z.ndim == 2 and snr.ndim == 1 and z.shape[1] % snr.numel() == 0:
        repeat = z.shape[1] // snr.numel()
        snr = snr.repeat_interleave(repeat)

    snr = snr.reshape(1, -1, *([1] * max(0, z.ndim - 2)))
    if z.ndim == 2:
        signal_power = torch.mean(torch.abs(z) ** 2, dim=0, keepdim=True)
    else:
        signal_power = torch.mean(torch.abs(z) ** 2, dim=(0, 2, 3), keepdim=True)
    return snr, signal_power


def awgn_channel(z: torch.Tensor, snr) -> torch.Tensor:
    snr = _as_snr_tensor(snr, z.device)
    if snr is None:
        return z

    snr, signal_power = _signal_power(z, snr)
    noise_power = signal_power / torch.pow(10.0, snr / 10.0)
    noise = torch.randn_like(z) * torch.sqrt(noise_power)
    return z + noise


def dynamic_awgn_channel(z: torch.Tensor, snr) -> torch.Tensor:
    return awgn_channel(z, snr)


def rayleigh_channel(z: torch.Tensor, snr) -> torch.Tensor:
    h = torch.randn_like(z)
    z_faded = h * z

    snr = _as_snr_tensor(snr, z.device)
    if snr is None:
        return z_faded

    snr, signal_power = _signal_power(z, snr)
    noise_power = signal_power / torch.pow(10.0, snr / 10.0)
    noise = torch.randn_like(z) * torch.sqrt(noise_power)
    return z_faded + noise


def get_channel(channel_type) -> ChannelFn:
    if channel_type is None or str(channel_type).lower() in {"none", "null"}:
        return identity_channel
    if channel_type == "awgn":
        return awgn_channel
    if channel_type == "dynamic_awgn":
        return dynamic_awgn_channel
    if channel_type == "rayleigh":
        return rayleigh_channel
    raise ValueError(f"Unsupported channel type: {channel_type}")
