#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .functional import awgn_channel


class AWGNChannel:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def forward(self, x, snr):
        return awgn_channel(x, snr)

    def forward_complex(self, x, snr):
        return awgn_channel(x, snr)
