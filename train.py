#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.hydra_argparse_compat import patch_hydra_argparse_help_py314


def _init_tensorboard(save_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("TensorBoard is not installed; continuing without it")
        return None

    tb_log_dir = os.path.join(save_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    logger.info(f"TensorBoard logging to {tb_log_dir}")
    return SummaryWriter(log_dir=tb_log_dir)


def _save_config_snapshot(cfg: DictConfig):
    config_save_dir = os.path.join(cfg.save_dir, "configs")
    os.makedirs(config_save_dir, exist_ok=True)

    resolved_config_path = os.path.join(config_save_dir, "resolved_config.yaml")
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Saved resolved config to {resolved_config_path}")

    original_config_path = os.path.join(os.path.dirname(__file__), "configs", "main.yaml")
    if os.path.exists(original_config_path):
        shutil.copy(original_config_path, os.path.join(config_save_dir, "main.yaml"))


def _create_scheduler(optimizer, cfg, max_epochs):
    import torch

    scheduler_cfg = getattr(cfg, "lr_scheduler", None)
    if scheduler_cfg is None:
        return None, 0

    scheduler_type = getattr(scheduler_cfg, "type", "none")
    warmup_epochs = int(getattr(scheduler_cfg, "warmup_epochs", 0))

    if scheduler_type == "none":
        return None, warmup_epochs
    if scheduler_type == "cosine":
        min_lr = getattr(scheduler_cfg, "min_lr", 1e-6)
        t_max = max(1, max_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=min_lr
        )
        return scheduler, warmup_epochs
    if scheduler_type == "reduce_on_plateau":
        factor = getattr(scheduler_cfg, "factor", 0.1)
        patience = getattr(scheduler_cfg, "patience", 10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        )
        return scheduler, warmup_epochs

    logger.warning(f"Unknown scheduler type: {scheduler_type}, using none")
    return None, warmup_epochs


class Lion:
    """Minimal Lion optimizer implementation.

    Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms",
    2023. This keeps the training entrypoint self-contained because PyTorch
    does not currently ship Lion in torch.optim.
    """

    def __new__(cls, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        import torch

        class _Lion(torch.optim.Optimizer):
            def __init__(self, params, lr, betas, weight_decay):
                if lr < 0:
                    raise ValueError(f"Invalid learning rate: {lr}")
                if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
                    raise ValueError(f"Invalid beta parameters: {betas}")
                if weight_decay < 0:
                    raise ValueError(f"Invalid weight_decay value: {weight_decay}")
                defaults = {
                    "lr": lr,
                    "betas": betas,
                    "weight_decay": weight_decay,
                }
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    lr = group["lr"]
                    beta1, beta2 = group["betas"]
                    weight_decay = group["weight_decay"]

                    for param in group["params"]:
                        if param.grad is None:
                            continue

                        grad = param.grad
                        if grad.is_sparse:
                            raise RuntimeError("Lion does not support sparse gradients")

                        if weight_decay != 0:
                            param.mul_(1 - lr * weight_decay)

                        state = self.state[param]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(param)

                        exp_avg = state["exp_avg"]
                        update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                        param.add_(update.sign(), alpha=-lr)
                        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                return loss

        return _Lion(params, lr=lr, betas=betas, weight_decay=weight_decay)


def _as_betas(value, default):
    if value is None:
        return default
    return tuple(float(item) for item in value)


def _create_optimizer(model, cfg):
    import torch

    optimizer_cfg = getattr(cfg, "optimizer", None)
    optimizer_type = str(getattr(optimizer_cfg, "type", "adam")).lower()
    learning_rate = float(getattr(cfg, "learning_rate"))
    weight_decay = float(getattr(optimizer_cfg, "weight_decay", 0.0))

    if optimizer_type == "adam":
        betas = _as_betas(getattr(optimizer_cfg, "betas", None), (0.9, 0.999))
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
    if optimizer_type == "adamw":
        betas = _as_betas(getattr(optimizer_cfg, "betas", None), (0.9, 0.999))
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
    if optimizer_type == "lion":
        betas = _as_betas(getattr(optimizer_cfg, "betas", None), (0.9, 0.99))
        return Lion(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unknown optimizer.type={optimizer_type}. Available: adam, adamw, lion"
    )


def _amp_settings(cfg, device):
    import torch

    precision = str(getattr(cfg, "precision", "32-true"))
    use_amp = device.type == "cuda" and ("mixed" in precision or "16" in precision)

    if not use_amp:
        return False, torch.float32, False
    if "bf16" in precision:
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _sample_snr(cfg, x, device):
    import torch

    channel_type = getattr(cfg, "channel_type", None)
    if channel_type is None or channel_type == "null":
        return torch.full((x.shape[0],), cfg.max_snr, device=device)

    if channel_type == "dynamic_awgn":
        from src.utils.snr_utils import generate_per_channel_snr

        num_channels = int(getattr(cfg.coder, "comm_channels", 96)) // 2
        strategy = getattr(cfg, "per_channel_snr_strategy", "random")
        return generate_per_channel_snr(
            num_channels=num_channels,
            min_snr=cfg.min_snr,
            max_snr=cfg.max_snr,
            strategy=strategy,
            device=device,
        )

    return torch.rand(x.shape[0], device=device) * (cfg.max_snr - cfg.min_snr) + cfg.min_snr


def _load_checkpoint_if_needed(cfg, model, optimizer, scheduler, device):
    import torch

    checkpoint_path = getattr(cfg, "checkpoint", None)
    if checkpoint_path is None:
        return 0
    if not os.path.isfile(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} not found")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if "lpips_loss" in key or ("loss" in key and "scaling_layer" in key):
            continue
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

    if getattr(cfg, "checkpoint_scheduler", False):
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1 if isinstance(checkpoint, dict) else 0
    logger.info(f"Loaded checkpoint from {checkpoint_path}; starting at epoch {start_epoch}")
    return start_epoch


def _log_scalars(writer, metrics, step):
    if writer is None:
        return
    for key, value in metrics.items():
        if key != "epoch":
            writer.add_scalar(key, value, step)


@hydra.main(version_base="1.3", config_path="configs", config_name="main")
def main(cfg: DictConfig):
    import torch

    from src.data_loader import get_dataloaders
    from src.model_loader import get_model
    from src.utils import configure_pytorch, print_system_info, setup_training_environment

    setup_training_environment()
    configure_pytorch()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print_system_info()
    logger.info(f"Single-device training on {device}")
    logger.info(f"Saving to {cfg.save_dir}")

    writer = _init_tensorboard(cfg.save_dir)
    _save_config_snapshot(cfg)

    model = get_model(cfg).to(device)

    if getattr(cfg, "compile", False):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    try:
        from torchinfo import summary

        input_size = (cfg.batch_size, 3, cfg.patch_size, cfg.patch_size)
        summary(model, input_size=input_size, depth=1)
    except Exception as exc:
        logger.warning(f"Skipping model summary: {exc}")

    train_loader, val_loader = get_dataloaders(cfg)

    optimizer = _create_optimizer(model, cfg)
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    scheduler, warmup_epochs = _create_scheduler(
        optimizer, cfg, max_epochs=int(cfg.max_epochs)
    )
    base_lr = getattr(cfg, "learning_rate")

    use_amp, amp_dtype, use_scaler = _amp_settings(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    logger.info(f"AMP enabled: {use_amp}, dtype: {amp_dtype}, scaler: {use_scaler}")

    gradient_clip_val = float(getattr(cfg, "gradient_clip_val", 0.0))
    start_epoch = _load_checkpoint_if_needed(cfg, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, int(cfg.max_epochs)):
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
            logger.info(f"Warmup epoch {epoch}, lr: {warmup_lr:.6f}")

        model.train()
        train_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.max_epochs}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            x = batch.to(device, non_blocking=True)
            snr = _sample_snr(cfg, x, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                total_loss, diff_loss, real_rate_loss, x_hat, mse_loss = model(x, snr=snr)
                psnr_val = model.psnr(x_hat.detach().clamp(0, 1), x).item()

            if use_scaler:
                scaler.scale(total_loss).backward()
                if gradient_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                optimizer.step()

            loss_val = total_loss.detach().item()
            train_loss_sum += loss_val
            num_batches += 1

            if batch_idx % 10 == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss_val:.4f}",
                        "mse": f"{mse_loss.item():.4f}",
                        "psnr": f"{psnr_val:.2f}",
                    }
                )
                metrics = {
                    "train/diff_loss": diff_loss.item(),
                    "train/pixel_loss": mse_loss.item(),
                    "train/rate_loss": (
                        real_rate_loss
                        if isinstance(real_rate_loss, (int, float))
                        else real_rate_loss.item()
                    ),
                    "train/loss": loss_val,
                    "train/psnr": psnr_val,
                    "epoch": epoch,
                }
                _log_scalars(writer, metrics, epoch * len(train_loader) + batch_idx)

        train_loss_avg = train_loss_sum / max(1, num_batches)
        logger.info(f"Epoch {epoch} train loss: {train_loss_avg:.4f}")

        model.eval()
        val_loss_sum = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_mse_sum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", dynamic_ncols=True):
                x = batch.to(device, non_blocking=True)
                snr = _sample_snr(cfg, x, device)

                with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp, dtype=amp_dtype
                ):
                    loss, _, _, x_hat, mse_loss = model(x, snr=snr)

                x_hat_clamped = x_hat.detach().clamp(0, 1)
                val_loss_sum += loss.item()
                val_psnr_sum += model.psnr(x_hat_clamped, x).item()
                val_ssim_sum += model.ssim(x_hat_clamped, x).item()
                val_mse_sum += mse_loss.item()
                num_val_batches += 1

        if num_val_batches == 0:
            logger.warning("Validation loader produced no batches; skipping scheduler and checkpoint")
            continue

        val_loss_avg = val_loss_sum / num_val_batches
        val_psnr_avg = val_psnr_sum / num_val_batches
        val_ssim_avg = val_ssim_sum / num_val_batches
        val_mse_avg = val_mse_sum / num_val_batches

        if scheduler is not None and epoch >= warmup_epochs:
            scheduler_type = getattr(getattr(cfg, "lr_scheduler", None), "type", "none")
            if scheduler_type == "reduce_on_plateau":
                scheduler.step(val_loss_avg)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch} lr: {current_lr:.6f}")
        logger.info(
            f"Epoch {epoch} val loss: {val_loss_avg:.4f}, "
            f"PSNR: {val_psnr_avg:.2f}, SSIM: {val_ssim_avg:.4f}"
        )

        epoch_metrics = {
            "epoch": epoch,
            "train/epoch_loss": train_loss_avg,
            "val/loss": val_loss_avg,
            "val/psnr": val_psnr_avg,
            "val/ssim": val_ssim_avg,
            "val/mse": val_mse_avg,
            "lr": current_lr,
        }
        _log_scalars(writer, epoch_metrics, epoch)

        ckpt_dir = os.path.join(cfg.save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        if getattr(cfg, "save_only_latest", False):
            ckpt_path = os.path.join(ckpt_dir, "latest.pth")
        else:
            ckpt_path = os.path.join(
                ckpt_dir, f"epoch={epoch:02d}-train_loss={train_loss_avg:.4f}.pth"
            )

        save_dict = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
        torch.save(save_dict, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    patch_hydra_argparse_help_py314()
    main()
