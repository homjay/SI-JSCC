#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .functional import rayleigh_channel


class RayleighChannel:
    """Rayleigh fading channel with AWGN noise."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def forward(self, x, snr):
        return rayleigh_channel(x, snr)

    def forward_complex(self, x, snr):
        return rayleigh_channel(x, snr)
