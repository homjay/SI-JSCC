#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ADJSCC (Attention Deep Joint Source Channel Coding) backbone.

Converted from TensorFlow implementation:
https://github.com/alexxu1988/ADJSCC

Reference:
J. Xu, B. Ai, W. Chen, A. Yang, P. Sun and M. Rodrigues,
"Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules,"
IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 4,
pp. 2315-2328, April 2022, doi: 10.1109/TCSVT.2021.3082521.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GDN(nn.Module):
    """
    Generalized Divisive Normalization (GDN) layer.

    GDN is a nonlinear normalization transform used in learned image compression.
    It normalizes activations based on learned relationships between channels.

    Reference: Ballé et al., "Density Modeling of Images Using a Generalized
    Normalization Transformation", ICLR 2016.

    Args:
        num_features: Number of input channels
        inverse: If True, compute IGDN (inverse GDN) instead of GDN
        beta_min: Minimum value for beta parameter (for numerical stability)
        gamma_init: Initial value for gamma diagonal
    """

    def __init__(self, num_features, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super().__init__()
        self.num_features = num_features
        self.inverse = inverse
        self.beta_min = beta_min

        # Beta parameter (per-channel bias)
        self.beta = nn.Parameter(torch.ones(num_features))

        # Gamma parameter (channel correlation matrix, stored as lower triangular)
        # Initialize as identity-like for stable training
        self.gamma = nn.Parameter(torch.eye(num_features) * gamma_init)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Compute squared activations
        x_sq = x**2  # (B, C, H, W)

        # Reshape for matrix multiplication: (B, H, W, C)
        x_sq_t = x_sq.permute(0, 2, 3, 1).contiguous()

        # Compute normalization factor: gamma @ x^2 + beta
        # gamma is (C, C), x_sq_t is (B, H, W, C)
        # Result: (B, H, W, C)
        gamma_sq = self.gamma**2  # Ensure non-negative
        norm = F.linear(x_sq_t, gamma_sq) + self.beta**2 + self.beta_min

        # Reshape back: (B, C, H, W)
        norm = norm.permute(0, 3, 1, 2).contiguous()

        if self.inverse:
            # IGDN: multiply by sqrt(norm)
            return x * torch.sqrt(norm)
        else:
            # GDN: divide by sqrt(norm)
            return x / torch.sqrt(norm)


class SignalConv2D(nn.Module):
    """
    Signal convolution layer that wraps Conv2d or ConvTranspose2d with GDN activation.

    This mimics tensorflow_compression.SignalConv2D behavior.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size (int or tuple)
        stride: Stride for downsampling (corr=True) or upsampling (corr=False)
        corr: If True, use correlation (Conv2d), else use convolution (ConvTranspose2d)
        use_gdn: If True, apply GDN/IGDN after convolution
        inverse_gdn: If True and use_gdn, apply IGDN instead of GDN
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        corr=True,
        use_gdn=True,
        inverse_gdn=False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        if corr:
            # Correlation (downsampling)
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            )
        else:
            # Transpose convolution (upsampling)
            output_padding = (stride[0] - 1, stride[1] - 1) if stride[0] > 1 else (0, 0)
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=True,
            )

        self.use_gdn = use_gdn
        if use_gdn:
            self.gdn = GDN(out_channels, inverse=inverse_gdn)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        if self.use_gdn:
            x = self.gdn(x)
        return x


class AFModule(nn.Module):
    """
    Attention Feature Module (AF Module) for SNR-adaptive feature modulation.

    Uses channel attention (SE-Net style) conditioned on SNR to adaptively
    rescale feature channels based on channel conditions.

    Args:
        num_channels: Number of input/output channels
        reduction: Channel reduction ratio for bottleneck
    """

    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.num_channels = num_channels

        # Global average pooling is done in forward

        # FC layers with SNR concatenation
        # Input: num_channels (from GAP) + 1 (SNR)
        self.fc1 = nn.Linear(num_channels + 1, num_channels // reduction)
        self.fc2 = nn.Linear(num_channels // reduction, num_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x, snr):
        """
        Args:
            x: Input features (B, C, H, W)
            snr: SNR values (B,) or (B, 1)

        Returns:
            Modulated features (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Global average pooling: (B, C)
        m = F.adaptive_avg_pool2d(x, 1).view(B, C)

        # Ensure SNR has correct shape: (B, 1)
        if snr.dim() == 0:
            snr = snr.unsqueeze(0).expand(B)
        if snr.dim() == 1:
            snr = snr.view(B, 1)

        # Concatenate with SNR: (B, C+1)
        m = torch.cat([m, snr], dim=1)

        # FC layers
        m = F.relu(self.fc1(m))
        m = torch.sigmoid(self.fc2(m))

        # Expand and apply attention: (B, C, 1, 1)
        m = m.view(B, C, 1, 1)

        return x * m


class GFREncoderModule(nn.Module):
    """
    GFR (GDN-based Feature Representation) Encoder Module.

    Consists of SignalConv2D with optional PReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None):
        super().__init__()
        self.conv = SignalConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            corr=True,
            use_gdn=True,
            inverse_gdn=False,
        )

        self.activation = None
        if activation == "prelu":
            self.activation = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class GFRDecoderModule(nn.Module):
    """
    GFR (GDN-based Feature Representation) Decoder Module.

    Consists of SignalConv2D (transpose) with optional PReLU/Sigmoid activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None):
        super().__init__()
        self.conv = SignalConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            corr=False,
            use_gdn=True,
            inverse_gdn=True,
        )

        self.activation = None
        if activation == "prelu":
            self.activation = nn.PReLU(num_parameters=out_channels)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# =============================================================================
# BDJSCC: Basic Deep JSCC (without attention)
# =============================================================================


class BDJSCCEnCoder(nn.Module):
    """
    Basic Deep JSCC Encoder (without attention).

    Architecture:
        - en1: 3 -> 256, 9x9, stride 2, GDN + PReLU
        - en2: 256 -> 256, 5x5, stride 2, GDN + PReLU
        - en3: 256 -> 256, 5x5, stride 1, GDN + PReLU
        - en4: 256 -> 256, 5x5, stride 1, GDN + PReLU
        - en5: 256 -> tcn, 5x5, stride 1, GDN only
    """

    def __init__(self, cfg, base_channels=256):
        super().__init__()
        self.cfg = cfg
        comm_channels = int(cfg.coder.comm_channels)
        base_channels = int(getattr(cfg.coder, "base_channels", base_channels))

        self.en1 = GFREncoderModule(3, base_channels, (9, 9), 2, "prelu")
        self.en2 = GFREncoderModule(base_channels, base_channels, (5, 5), 2, "prelu")
        self.en3 = GFREncoderModule(base_channels, base_channels, (5, 5), 1, "prelu")
        self.en4 = GFREncoderModule(base_channels, base_channels, (5, 5), 1, "prelu")
        self.en5 = GFREncoderModule(base_channels, comm_channels, (5, 5), 1, None)

    def forward(self, x, snr=None, **kwargs):
        x = self.en1(x)
        x = self.en2(x)
        x = self.en3(x)
        x = self.en4(x)
        x = self.en5(x)
        return {"x": x}


class BDJSCCDeCoder(nn.Module):
    """
    Basic Deep JSCC Decoder (without attention).

    Architecture:
        - de1: tcn -> 256, 5x5, stride 1, IGDN + PReLU
        - de2: 256 -> 256, 5x5, stride 1, IGDN + PReLU
        - de3: 256 -> 256, 5x5, stride 1, IGDN + PReLU
        - de4: 256 -> 256, 5x5, stride 2, IGDN + PReLU
        - de5: 256 -> 3, 9x9, stride 2, IGDN + Sigmoid
    """

    def __init__(self, cfg, base_channels=256):
        super().__init__()
        self.cfg = cfg
        comm_channels = int(cfg.coder.comm_channels)
        base_channels = int(getattr(cfg.coder, "base_channels", base_channels))

        self.de1 = GFRDecoderModule(comm_channels, base_channels, (5, 5), 1, "prelu")
        self.de2 = GFRDecoderModule(base_channels, base_channels, (5, 5), 1, "prelu")
        self.de3 = GFRDecoderModule(base_channels, base_channels, (5, 5), 1, "prelu")
        self.de4 = GFRDecoderModule(base_channels, base_channels, (5, 5), 2, "prelu")
        self.de5 = GFRDecoderModule(base_channels, 3, (9, 9), 2, "sigmoid")

    def forward(self, x, snr=None, **kwargs):
        x = self.de1(x)
        x = self.de2(x)
        x = self.de3(x)
        x = self.de4(x)
        x = self.de5(x)
        return {"x": x}


# =============================================================================
# ADJSCC: Attention Deep JSCC
# =============================================================================


class ADJSCCEnCoder(nn.Module):
    """
    Attention Deep JSCC Encoder.

    Same as BDJSCCEnCoder but with AF_Module after each layer (except last).
    The attention module uses SNR information to adaptively modulate features.
    """

    def __init__(self, cfg, base_channels=256):
        super().__init__()
        self.cfg = cfg
        comm_channels = int(cfg.coder.comm_channels)
        base_channels = int(getattr(cfg.coder, "base_channels", base_channels))
        reduction = int(getattr(cfg.coder, "attention_reduction", 16))

        self.en1 = GFREncoderModule(3, base_channels, (9, 9), 2, "prelu")
        self.af1 = AFModule(base_channels, reduction)

        self.en2 = GFREncoderModule(base_channels, base_channels, (5, 5), 2, "prelu")
        self.af2 = AFModule(base_channels, reduction)

        self.en3 = GFREncoderModule(base_channels, base_channels, (5, 5), 1, "prelu")
        self.af3 = AFModule(base_channels, reduction)

        self.en4 = GFREncoderModule(base_channels, base_channels, (5, 5), 1, "prelu")
        self.af4 = AFModule(base_channels, reduction)

        self.en5 = GFREncoderModule(base_channels, comm_channels, (5, 5), 1, None)

    def forward(self, x, snr=None, **kwargs):
        # Default SNR if not provided
        if snr is None:
            snr = torch.zeros(x.size(0), device=x.device)

        x = self.en1(x)
        x = self.af1(x, snr)

        x = self.en2(x)
        x = self.af2(x, snr)

        x = self.en3(x)
        x = self.af3(x, snr)

        x = self.en4(x)
        x = self.af4(x, snr)

        x = self.en5(x)
        return {"x": x}


class ADJSCCDeCoder(nn.Module):
    """
    Attention Deep JSCC Decoder.

    Same as BDJSCCDeCoder but with AF_Module after each layer (except last).
    The attention module uses SNR information to adaptively modulate features.
    """

    def __init__(self, cfg, base_channels=256):
        super().__init__()
        self.cfg = cfg
        comm_channels = int(cfg.coder.comm_channels)
        base_channels = int(getattr(cfg.coder, "base_channels", base_channels))
        reduction = int(getattr(cfg.coder, "attention_reduction", 16))

        self.de1 = GFRDecoderModule(comm_channels, base_channels, (5, 5), 1, "prelu")
        self.af1 = AFModule(base_channels, reduction)

        self.de2 = GFRDecoderModule(base_channels, base_channels, (5, 5), 1, "prelu")
        self.af2 = AFModule(base_channels, reduction)

        self.de3 = GFRDecoderModule(base_channels, base_channels, (5, 5), 1, "prelu")
        self.af3 = AFModule(base_channels, reduction)

        self.de4 = GFRDecoderModule(base_channels, base_channels, (5, 5), 2, "prelu")
        self.af4 = AFModule(base_channels, reduction)

        self.de5 = GFRDecoderModule(base_channels, 3, (9, 9), 2, "sigmoid")

    def forward(self, x, snr=None, **kwargs):
        # Default SNR if not provided
        if snr is None:
            snr = torch.zeros(x.size(0), device=x.device)

        x = self.de1(x)
        x = self.af1(x, snr)

        x = self.de2(x)
        x = self.af2(x, snr)

        x = self.de3(x)
        x = self.af3(x, snr)

        x = self.de4(x)
        x = self.af4(x, snr)

        x = self.de5(x)
        return {"x": x}
