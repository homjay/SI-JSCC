#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2026 Jay
# Licensed under the MIT License. See LICENSE in the project root.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerLayerNorm(nn.Module):
    """
    Power Layer Normalization that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    This implementation uses a power normalization approach, which is a variant of layer normalization.
    It normalizes the input tensor by subtracting the mean and dividing by the standard deviation,
    followed by a power transformation to enhance the representation of the normalized features.
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = torch.ones(normalized_shape)
        self.bias = torch.zeros(normalized_shape)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return x
