#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single-file CBJSCC backbone.

This file contains the default CBJSCC encoder/decoder and its local feature
blocks so the main model can be inspected without jumping across several
backbone files.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models.utils import conv, deconv


def _make_pair(value):
    if isinstance(value, int):
        return (value, value)
    return value


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU(inplace)
    if act_type == "lrelu":
        return nn.LeakyReLU(neg_slope, inplace)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    raise NotImplementedError(f"activation layer [{act_type}] is not found")


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]

    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ESA(nn.Module):
    """Enhanced Spatial Attention block used inside RLFB."""

    def __init__(self, esa_channels, n_feats, conv_fn):
        super().__init__()
        self.conv1 = conv_fn(n_feats, esa_channels, kernel_size=1)
        self.conv_f = conv_fn(esa_channels, esa_channels, kernel_size=1)
        self.conv2 = conv_fn(esa_channels, esa_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv_fn(esa_channels, esa_channels, kernel_size=3, padding=1)
        self.conv4 = conv_fn(esa_channels, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_dtype = x.dtype
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=5, stride=3, padding=2)
        c3 = self.conv3(v_max).to(dtype=torch.float32)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        c3 = c3.to(dtype=input_dtype)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        return x * self.sigmoid(c4)


class IRLFB(nn.Module):
    """Residual Local Feature Block."""

    def __init__(self, in_channels, mid_channels=None, out_channels=None, esa_channels=16):
        super().__init__()
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels

        self.c1_r = conv_layer(in_channels, in_channels, 3)
        self.c2_r = conv_layer(in_channels, mid_channels, 1)
        self.c3_r = conv_layer(mid_channels, in_channels, 1)
        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
        self.act = activation("lrelu", neg_slope=0.05)

    def forward(self, x):
        out = self.act(self.c1_r(x))
        out = self.act(self.c2_r(out))
        out = self.act(self.c3_r(out))
        out = out + x
        out = self.c5(out)
        return self.esa(out)


class AFModule(nn.Module):
    """SNR-conditioned channel modulation module."""

    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.factor_liner_1 = nn.Linear(in_features=dim + 1, out_features=dim)
        self.factor_activate = nn.ReLU()
        self.factor_liner_2 = nn.Linear(in_features=dim, out_features=dim)
        self.factor_sigmoid = nn.Sigmoid()

    def forward(self, x, snr):
        if not isinstance(snr, torch.Tensor):
            snr = torch.tensor(snr, device=x.device)
        snr = snr.to(x.device)
        if snr.ndim == 0:
            snr = snr.expand(x.shape[0])
        if snr.ndim > 1:
            snr = snr.flatten()[: x.shape[0]]

        z_mean = self.pool(x)
        inter_shape = z_mean.shape
        z_mean = z_mean.view(z_mean.shape[0], -1)
        snr_prior = snr.view(-1, 1)
        z_cat = torch.cat((z_mean, snr_prior), dim=-1)
        factor = self.factor_liner_1(z_cat)
        factor = self.factor_activate(factor)
        factor = self.factor_liner_2(factor)
        factor = self.factor_sigmoid(factor).view(inter_shape)
        return (x * factor).contiguous()


class CBJSCCEnCoder(nn.Module):
    def __init__(self, cfg, snr_prior=False):
        super().__init__()
        num_filters = int(getattr(cfg.coder, "num_filters", 192))
        comm_channels = int(cfg.coder.comm_channels)
        snr_prior = bool(getattr(cfg.coder, "snr_prior", snr_prior))

        self.conv1 = conv(3, num_filters)
        self.block1 = IRLFB(num_filters, num_filters * 2, num_filters)
        self.block2 = IRLFB(num_filters, num_filters * 2, num_filters)
        self.conv2 = conv(num_filters, num_filters)
        self.block4 = IRLFB(num_filters, num_filters * 4, num_filters)
        self.block5 = IRLFB(num_filters, num_filters * 4, num_filters)
        self.block6 = IRLFB(num_filters, comm_channels, comm_channels)
        self.snr_prior = snr_prior
        if self.snr_prior:
            self.af_module1 = AFModule(num_filters)
            self.af_module2 = AFModule(comm_channels)

    def forward(self, x, snr=None):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv2(x)
        x = self.block4(x)
        x = self.block5(x)
        if self.snr_prior:
            x = self.af_module1(x, snr)
        x = self.block6(x)
        if self.snr_prior:
            x = self.af_module2(x, snr)
        return {"x": x}


class CBJSCCDeCoder(nn.Module):
    def __init__(self, cfg, snr_prior=False):
        super().__init__()
        num_filters = int(getattr(cfg.coder, "num_filters", 192))
        comm_channels = int(cfg.coder.comm_channels)
        snr_prior = bool(getattr(cfg.coder, "snr_prior", snr_prior))

        self.snr_prior = snr_prior
        if self.snr_prior:
            self.af_module1 = AFModule(comm_channels)
            self.af_module2 = AFModule(num_filters)
        self.block1 = IRLFB(comm_channels, num_filters * 4, num_filters)
        self.block2 = IRLFB(num_filters, num_filters * 4, num_filters)
        self.block3 = IRLFB(num_filters, num_filters * 4, num_filters)
        self.conv2 = deconv(num_filters, num_filters)
        self.block5 = IRLFB(num_filters, num_filters * 2, num_filters)
        self.block6 = IRLFB(num_filters, num_filters * 2, num_filters)
        self.conv3 = deconv(num_filters, 3)

    def forward(self, x, snr=None):
        if self.snr_prior:
            x = self.af_module1(x, snr)
        x = self.block1(x)
        if self.snr_prior:
            x = self.af_module2(x, snr)
        x = self.block2(x)
        x = self.block3(x)
        x = self.conv2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv3(x)
        return {"x": x}
