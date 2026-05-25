#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import torch
import torch.nn as nn
from loguru import logger
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from .channels import get_channel, identity_channel
from .utils.l1loss import L1CharbonnierLoss


class JSCCAutoEncoder(nn.Module):
    """End-to-end JSCC model: image encoder -> channel -> image decoder."""

    def __init__(self, cfg, encoder, decoder, compile=False):
        super().__init__()
        self.cfg = cfg
        self.encoder = torch.compile(encoder) if compile else encoder
        self.decoder = torch.compile(decoder) if compile else decoder

        self.channel_type = getattr(cfg, "channel_type", None)
        self.channel = get_channel(self.channel_type)
        self.pass_snr = getattr(cfg.coder, "pass_snr", False)
        self.min_snr = cfg.min_snr
        self.max_snr = cfg.max_snr
        self.enable_rate_loss = getattr(cfg, "enable_rate_loss", False)

        comm_channels = int(cfg.coder.comm_channels)
        self.power_norm_type = str(
            getattr(cfg.coder, "power_norm", getattr(cfg, "power_norm", "batchnorm"))
        ).lower()
        self.complex_pairing = str(getattr(cfg.coder, "complex_pairing", "interleave"))
        if self.power_norm_type == "batchnorm":
            self.power_norm_enc = nn.BatchNorm2d(num_features=comm_channels, eps=1e-5)
            self.power_norm_dec = nn.BatchNorm2d(num_features=comm_channels, eps=1e-5)
        elif self.power_norm_type in {"average_power", "avg_power"}:
            self.power_norm_enc = nn.Identity()
            self.power_norm_dec = nn.Identity()
        else:
            raise ValueError(
                "Unsupported power_norm. Available: batchnorm, average_power"
            )
        self.flatten = nn.Flatten()

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.charbonnier_loss = L1CharbonnierLoss()
        self.lpips_loss = None

        self.psnr = PeakSignalNoiseRatio(data_range=(0, 1))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        if self._needs_lpips():
            self._init_lpips_loss()

    def _needs_lpips(self) -> bool:
        loss_type = getattr(self.cfg, "loss_type", "l2")
        adaptive_cfg = getattr(self.cfg, "loss_adaptive", {})
        adaptive_enabled = (
            adaptive_cfg.get("enabled", False)
            if hasattr(adaptive_cfg, "get")
            else getattr(adaptive_cfg, "enabled", False)
        )
        return loss_type == "lpips" or adaptive_enabled

    def _init_lpips_loss(self):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS loss requires the 'lpips' package. Install it or use "
                "loss_type=l2/l1/l1_charbonnier."
            ) from exc

        self.lpips_loss = lpips.LPIPS(pretrained=True, net="squeeze", eval_mode=True)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def _get_lpips_loss(self):
        if self.lpips_loss is None:
            self._init_lpips_loss()
        return self.lpips_loss

    def _real_to_complex(self, x):
        if self.complex_pairing == "split":
            x_flat = self.flatten(x).contiguous()
            dim_z = x_flat.shape[1] // 2
            return torch.complex(x_flat[:, :dim_z], x_flat[:, dim_z : 2 * dim_z])

        x_real = self.flatten(x[:, ::2, ...]).contiguous()
        x_imag = self.flatten(x[:, 1::2, ...]).contiguous()
        return torch.complex(x_real, x_imag)

    def _complex_to_real(self, x, shape):
        if self.complex_pairing == "split":
            x_flat = torch.cat([x.real, x.imag], dim=1)
            return x_flat.view(shape).contiguous()

        x_real = x.real.view(shape[0], -1, shape[2], shape[3])
        x_imag = x.imag.view(shape[0], -1, shape[2], shape[3])
        x = torch.stack([x_real, x_imag], dim=2)
        return x.view(shape).contiguous()

    def _normalize_average_power(self, z):
        dims = tuple(range(1, z.ndim))
        power = torch.mean(torch.abs(z) ** 2, dim=dims, keepdim=True)
        return z / torch.sqrt(power + 1e-8)

    def _preprocess_signal(self, z, freq_meta=None):
        self.inter_type = z.dtype
        self.inter_shape = z.shape
        self.inter_meta = freq_meta
        z = z.to(torch.float32)

        if freq_meta and freq_meta.get("packed_freq", False):
            z = torch.view_as_complex(z)
        else:
            z = self._real_to_complex(z)

        if self.power_norm_type in {"average_power", "avg_power"}:
            return self._normalize_average_power(z)
        return z * 0.70710678

    def _postprocess_signal(self, z):
        if self.inter_meta and self.inter_meta.get("packed_freq", False):
            z = torch.view_as_real(z).view(self.inter_shape)
        else:
            z = self._complex_to_real(z, self.inter_shape)
        if self.power_norm_type in {"average_power", "avg_power"}:
            return z.to(dtype=self.inter_type)
        return (z * 1.41421356).to(dtype=self.inter_type)

    def _run_encoder(self, x, snr=None):
        if self.cfg.return_features:
            output, rate_loss = (
                self.encoder(x, snr=snr) if self.pass_snr else self.encoder(x)
            )
            return output[-1], rate_loss, None

        output = self.encoder(x, snr=snr) if self.pass_snr else self.encoder(x)
        rate_loss = 0
        if isinstance(output, tuple):
            output, rate_loss = output

        if isinstance(output, dict):
            return output["x"], rate_loss, output.get("freq_meta")

        return output, rate_loss, None

    def _run_decoder(self, z_hat, snr=None, freq_meta=None):
        decoder_kwargs = {"snr": snr}
        if freq_meta is not None:
            decoder_kwargs["freq_meta"] = freq_meta
        output = self.decoder(z_hat, **decoder_kwargs)
        return output["x"] if isinstance(output, dict) else output

    def progressive_masking(self, x, mask=None, stage_dim=1):
        _, channels, _, _ = x.shape
        stage_num = channels // stage_dim
        min_stage = max(1, min(getattr(self.cfg, "min_stage", 1), stage_num))
        max_stage = max(
            min_stage, min(getattr(self.cfg, "max_stage", stage_num), stage_num)
        )

        if mask is None:
            mask_channel = random.randint(min_stage, max_stage) * stage_dim
            mask = torch.zeros_like(x, dtype=torch.float32)
            mask[:, :mask_channel, ...] = 1.0
            keep_ratio = mask_channel / channels
        else:
            keep_ratio = mask.sum() / mask.numel()

        x_masked = x * mask
        if getattr(self.cfg, "progressive_masking_compensation", False):
            x_masked = x_masked / (keep_ratio + 1e-8)
        return x_masked, keep_ratio

    def progressive_masking_token(self, x, mask=None, min_tokens=None):
        batch, channels, height, width = x.shape
        total_tokens = channels * height * width
        min_tokens = min_tokens or getattr(self.cfg, "min_tokens", 256)
        min_tokens = max(1, min(min_tokens, total_tokens))

        if mask is None:
            num_tokens_to_keep = random.randint(min_tokens, total_tokens)
            x_flat = x.view(batch, -1)
            mask_flat = torch.zeros_like(x_flat, dtype=torch.float32)
            mask_flat[:, :num_tokens_to_keep] = 1.0
            x_masked = (x_flat * mask_flat).view(batch, channels, height, width)
            keep_ratio = num_tokens_to_keep / total_tokens
        else:
            x_masked = x * mask
            keep_ratio = mask.sum() / mask.numel()

        if getattr(self.cfg, "progressive_masking_compensation", False):
            x_masked = x_masked / (keep_ratio + 1e-8)
        return x_masked.contiguous(), keep_ratio

    def _apply_progressive_masking(self, z_hat):
        if not self.cfg.progressive_masking:
            return z_hat, 1.0

        masking_mode = getattr(self.cfg, "progressive_masking_mode", "channel")
        if masking_mode == "token":
            return self.progressive_masking_token(z_hat)
        return self.progressive_masking(z_hat)

    def _adaptive_loss(self, x_hat, x, mse_loss, snr, keep_ratio):
        adaptive_cfg = getattr(self.cfg, "loss_adaptive", {})

        if snr is None:
            snr_val = torch.tensor(self.max_snr, device=x.device)
        elif torch.is_tensor(snr):
            snr_val = snr.to(device=x.device)
        else:
            snr_val = torch.tensor(snr, dtype=torch.float32, device=x.device)

        if snr_val.ndim > 0:
            snr_val = snr_val.mean()

        norm_snr = (snr_val - self.min_snr) / (self.max_snr - self.min_snr + 1e-8)
        norm_snr = torch.clamp(norm_snr, 0.0, 1.0)
        score = adaptive_cfg.get("snr_weight", 0.5) * norm_snr + adaptive_cfg.get(
            "rate_weight", 0.5
        ) * keep_ratio

        lpips_max = adaptive_cfg.get("lpips_weight_max", 1.0)
        lpips_min = adaptive_cfg.get("lpips_weight_min", 0.1)
        pixel_max = adaptive_cfg.get("pixel_weight_max", 10.0)
        pixel_min = adaptive_cfg.get("pixel_weight_min", 1.0)

        loss_lpips = self._get_lpips_loss()(x_hat.float(), x.float())
        if loss_lpips.ndim > 0:
            loss_lpips = loss_lpips.mean()

        lpips_weight = lpips_max - score * (lpips_max - lpips_min)
        pixel_weight = pixel_min + score * (pixel_max - pixel_min)
        return lpips_weight * loss_lpips + pixel_weight * mse_loss

    def _model_loss(self, x_hat, x, snr, keep_ratio):
        with torch.autocast(device_type=x.device.type, enabled=False):
            mse_loss = self.mse_loss(x_hat.float(), x.float())
            if mse_loss.ndim > 0:
                mse_loss = mse_loss.mean()

            adaptive_cfg = getattr(self.cfg, "loss_adaptive", {})
            adaptive_enabled = (
                adaptive_cfg.get("enabled", False)
                if hasattr(adaptive_cfg, "get")
                else getattr(adaptive_cfg, "enabled", False)
            )
            if adaptive_enabled:
                return self._adaptive_loss(x_hat, x, mse_loss, snr, keep_ratio), mse_loss

            loss_type = getattr(self.cfg, "loss_type", "l2")
            if loss_type in {"mse", "l2"}:
                loss = mse_loss
            elif loss_type == "l1":
                loss = self.l1_loss(x_hat.float(), x.float())
            elif loss_type == "l1_charbonnier":
                loss = self.charbonnier_loss(x_hat.float(), x.float())
            elif loss_type == "lpips":
                loss = self._get_lpips_loss()(x_hat.float(), x.float())
            else:
                logger.warning(f"Unknown loss_type={loss_type}; falling back to MSE")
                loss = mse_loss

            if loss.ndim > 0:
                loss = loss.mean()
            return loss, mse_loss

    def forward(self, x, snr=None):
        z, rate_loss, freq_meta = self._run_encoder(x, snr=snr)
        z = self.power_norm_enc(z)
        z = self._preprocess_signal(z, freq_meta)

        z_hat = self.channel(z, snr)
        decoder_snr = 999 if self.channel is identity_channel and snr is None else snr

        z_hat = self._postprocess_signal(z_hat)
        z_hat, keep_ratio = self._apply_progressive_masking(z_hat)
        z_hat = self.power_norm_dec(z_hat)

        x_hat = self._run_decoder(z_hat, snr=decoder_snr, freq_meta=freq_meta)
        model_loss, mse_loss = self._model_loss(x_hat, x, decoder_snr, keep_ratio)
        loss = model_loss + rate_loss if self.enable_rate_loss else model_loss
        return loss, model_loss, rate_loss, x_hat, mse_loss

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if "lpips_loss" in key or ("loss" in key and "scaling_layer" in key):
                continue
            cleaned_state_dict[key[7:] if key.startswith("module.") else key] = value

        missing_keys, unexpected_keys = self.load_state_dict(
            cleaned_state_dict, strict=False
        )
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        logger.info(f"Loaded checkpoint from {path}")
