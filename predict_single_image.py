#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision import transforms

from src.model_loader import get_model
from src.utils import configure_pytorch


def load_config(config_path):
    config_dir = os.path.dirname(config_path) or "configs"
    config_name = os.path.basename(config_path)
    if config_name.endswith(".yaml"):
        config_name = config_name[:-5]

    if os.path.isabs(config_dir):
        config_dir = os.path.relpath(config_dir, os.getcwd())

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=config_dir):
        return compose(config_name=config_name)


def find_latest_checkpoint(path):
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


def process_image(image_path, device, crop_size=None):
    image = Image.open(image_path).convert("RGB")
    if crop_size is not None:
        transform = transforms.Compose(
            [transforms.CenterCrop(crop_size), transforms.ToTensor()]
        )
    else:
        width, height = image.size
        new_width = max(64, (width // 64) * 64)
        new_height = max(64, (height // 64) * 64)
        if (new_width, new_height) != (width, height):
            logger.warning(
                f"Resizing image from {(width, height)} to "
                f"{(new_width, new_height)} for model downsampling."
            )
            image = image.resize((new_width, new_height), Image.BICUBIC)
        transform = transforms.ToTensor()

    return transform(image).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description="Run CBJSCC on a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--config", default="configs/main.yaml", help="Hydra config")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint file or directory containing .pth checkpoints",
    )
    parser.add_argument("--snr", type=float, default=10.0, help="SNR in dB")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    args = parser.parse_args()

    configure_pytorch()
    device = torch.device(args.device)

    cfg = load_config(args.config)
    model = get_model(cfg).to(device)
    model.load_checkpoint(find_latest_checkpoint(args.checkpoint))
    model.eval()

    image = process_image(args.image, device, args.crop_size)
    snr = torch.full((image.shape[0],), args.snr, device=device)

    with torch.no_grad():
        _, _, _, reconstructed, _ = model(image, snr=snr)
        reconstructed = reconstructed.clamp(0.0, 1.0)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    logger.info(f"PSNR: {psnr(reconstructed, image).item():.4f} dB")

    output = transforms.ToPILImage()(reconstructed.squeeze(0).cpu())
    output.save(args.output)
    logger.info(f"Saved reconstruction to {args.output}")


if __name__ == "__main__":
    main()
