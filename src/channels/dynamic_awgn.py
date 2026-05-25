#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .functional import dynamic_awgn_channel


class DynamicAWGNChannel:
    """AWGN channel with scalar, per-batch, or per-channel SNR support."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def forward(self, x, snr):
        return dynamic_awgn_channel(x, snr)

    def forward_complex(self, x, snr):
        return dynamic_awgn_channel(x, snr)
