#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

from src.data.single_path import SinglePathDatasetSpeedUP, get_transforms
from src.model_loader import get_model
from src.utils import configure_pytorch


def find_latest_checkpoint(path) -> str:
    root = Path(path)
    if root.is_file():
        return str(root)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {root}")

    checkpoints = sorted(
        root.rglob("*.pth"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoint files found under {root}")
    return str(checkpoints[0])


def load_config(project_path, config_path):
    candidates = []
    if config_path is not None:
        candidates.append(Path(config_path))
    if project_path is not None:
        project = Path(project_path)
        candidates.extend(
            [
                project / "configs" / "resolved_config.yaml",
                project / ".hydra" / "config.yaml",
                project / "config.yaml",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Loading config from {candidate}")
            return OmegaConf.load(candidate)

    raise FileNotFoundError(
        "No config found. Pass --config or use a training run directory with "
        "configs/resolved_config.yaml."
    )


def parse_snrs(value, cfg):
    if value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]

    min_snr = int(getattr(cfg, "min_snr", 1))
    max_snr = int(getattr(cfg, "max_snr", 13))
    gap_snr = int(getattr(cfg, "gap_snr", 2))
    return list(range(min_snr, max_snr + 1, gap_snr))


def make_dataloader(dataset_path, patch_size, batch_size, crop, num_workers):
    _, test_transforms = get_transforms(patch_size=patch_size, crop=crop)
    dataset = SinglePathDatasetSpeedUP(data_dir=dataset_path, transform=test_transforms)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_model(model, dataloader, device, snr_list, lpips=False):
    model.eval()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = (
        LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        if lpips
        else None
    )

    metrics = {
        snr: {"psnr": [], "ssim": [], "loss": [], "lpips": []} for snr in snr_list
    }

    for batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
        x = batch.to(device, non_blocking=True)

        for snr in snr_list:
            snr_tensor = torch.full((x.shape[0],), float(snr), device=device)
            with torch.no_grad():
                loss, _, _, x_hat, _ = model(x=x, snr=snr_tensor)

            x_hat = x_hat.clamp(0.0, 1.0)
            x_ref = x.clamp(0.0, 1.0)
            metrics[snr]["psnr"].append(psnr_metric(x_hat, x_ref).item())
            metrics[snr]["ssim"].append(ssim_metric(x_hat, x_ref).item())
            metrics[snr]["loss"].append(loss.item())
            if lpips_metric is not None:
                metrics[snr]["lpips"].append(lpips_metric(x_hat, x_ref).item())

    rows = []
    for snr in snr_list:
        row = {
            "snr": snr,
            "psnr": float(np.mean(metrics[snr]["psnr"])),
            "ssim": float(np.mean(metrics[snr]["ssim"])),
            "loss": float(np.mean(metrics[snr]["loss"])),
        }
        if lpips_metric is not None:
            row["lpips"] = float(np.mean(metrics[snr]["lpips"]))
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CBJSCC checkpoint")
    parser.add_argument("--dataset", required=True, help="Evaluation image directory")
    parser.add_argument("--project", default=None, help="Training run directory")
    parser.add_argument("--config", default=None, help="Path to resolved config YAML")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file or directory. Defaults to <project>/checkpoints.",
    )
    parser.add_argument("--snrs", default=None, help="Comma-separated SNR list")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Evaluation batch size. Use 1 for full-resolution datasets with "
            "mixed image sizes such as Kodak; larger values require --crop or "
            "uniform image dimensions."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop", action="store_true", help="Center-crop evaluation images")
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS")
    parser.add_argument(
        "--channel-type",
        default=None,
        choices=["awgn", "dynamic_awgn", "rayleigh", "null", "none"],
        help="Override channel type from the config",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument("--output", default=None, help="CSV output path")
    args = parser.parse_args()

    configure_pytorch()
    cfg = load_config(args.project, args.config)
    if args.channel_type is not None:
        cfg.channel_type = args.channel_type

    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.project is None:
            raise ValueError("Pass --checkpoint or --project")
        checkpoint = Path(args.project) / "checkpoints"
    checkpoint = find_latest_checkpoint(checkpoint)

    device = torch.device(args.device)
    model = get_model(cfg).to(device)
    model.load_checkpoint(checkpoint)

    dataloader = make_dataloader(
        dataset_path=args.dataset,
        patch_size=int(getattr(cfg, "patch_size", 128)),
        batch_size=args.batch_size,
        crop=args.crop,
        num_workers=args.num_workers,
    )
    snr_list = parse_snrs(args.snrs, cfg)
    logger.info(f"Evaluating {checkpoint} at SNRs: {snr_list}")

    results = evaluate_model(model, dataloader, device, snr_list, lpips=args.lpips)
    print(results.to_string(index=False))

    output = args.output
    if output is None:
        base_dir = Path(args.project) / "evaluation" if args.project else Path(".")
        base_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(args.dataset).name
        output = base_dir / f"{dataset_name}_metrics.csv"
    else:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output, index=False)
    logger.info(f"Saved results to {output}")


if __name__ == "__main__":
    main()
