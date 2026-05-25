#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DeepJSCC backbone adapted to the JSCC-AE encoder/decoder interface.
Channel simulation and power normalization are handled in JSCCAutoEncoder.
"""

import torch
import torch.nn as nn


def _get_activation(name):
    if name is None:
        return nn.Identity()
    name = str(name).lower()
    if name in ("identity", "none", "linear"):
        return nn.Identity()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    if name == "prelu":
        return nn.PReLU()
    raise ValueError(f"Unsupported activation: {name}")


def _init_conv_weight(conv, activation):
    if isinstance(activation, (nn.PReLU, nn.ReLU, nn.LeakyReLU)):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="leaky_relu")
    else:
        nn.init.xavier_normal_(conv.weight)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)


class _ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation="prelu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.act = _get_activation(activation)
        _init_conv_weight(self.conv, self.act)

    def forward(self, x):
        return self.act(self.conv(x))


class _TransConvAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
        activation="prelu",
    ):
        super().__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.act = _get_activation(activation)
        _init_conv_weight(self.transconv, self.act)

    def forward(self, x):
        return self.act(self.transconv(x))


class DeepJSCCEnCoder(nn.Module):
    def __init__(self, cfg, base_channels=32):
        super().__init__()
        self.cfg = cfg
        comm_channels = int(cfg.coder.comm_channels)
        base_channels = int(getattr(cfg.coder, "base_channels", base_channels))
        kernel_size = int(getattr(cfg.coder, "kernel_size", 5))
        padding = kernel_size // 2
        stem_channels = max(1, base_channels // 2)

        self.conv1 = _ConvAct(3, stem_channels, kernel_size, stride=2, padding=padding)
        self.conv2 = _ConvAct(stem_channels, base_channels, kernel_size, stride=2, padding=padding)
        self.conv3 = _ConvAct(base_channels, base_channels, kernel_size, padding=padding)
        self.conv4 = _ConvAct(base_channels, base_channels, kernel_size, padding=padding)
        self.conv5 = _ConvAct(base_channels, comm_channels, kernel_size, padding=padding)

    def forward(self, x, snr=None, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return {"x": x}


class DeepJSCCDeCoder(nn.Module):
    def __init__(self, cfg, base_channels=32):
        super().__init__()
        self.cfg = cfg
        comm_channels = int(cfg.coder.comm_channels)
        base_channels = int(getattr(cfg.coder, "base_channels", base_channels))
        kernel_size = int(getattr(cfg.coder, "kernel_size", 5))
        padding = kernel_size // 2
        stem_channels = max(1, base_channels // 2)
        output_activation = getattr(cfg.coder, "output_activation", "sigmoid")

        self.tconv1 = _TransConvAct(
            comm_channels, base_channels, kernel_size, stride=1, padding=padding
        )
        self.tconv2 = _TransConvAct(
            base_channels, base_channels, kernel_size, stride=1, padding=padding
        )
        self.tconv3 = _TransConvAct(
            base_channels, base_channels, kernel_size, stride=1, padding=padding
        )
        self.tconv4 = _TransConvAct(
            base_channels,
            stem_channels,
            kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
        )
        self.tconv5 = _TransConvAct(
            stem_channels,
            3,
            kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            activation=output_activation,
        )

    def forward(self, x, snr=None, **kwargs):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return {"x": x}

