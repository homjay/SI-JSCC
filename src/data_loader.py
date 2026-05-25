#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from loguru import logger
from torch.utils.data import DataLoader

from .data.single_path import SinglePathDatasetSpeedUP, get_transforms


def _worker_kwargs(num_workers):
    if num_workers <= 0:
        return {"num_workers": 0}
    return {
        "num_workers": num_workers,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }


def get_dataloaders(
    cfg,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
):
    logger.info("start loading data")
    if num_workers == 0:
        num_workers = int(cfg.num_workers)

    train_transforms, test_transforms = get_transforms(cfg.patch_size)
    train_dataset = SinglePathDatasetSpeedUP(
        data_dir=cfg.train_dataset.path,
        transform=train_transforms,
    )
    val_dataset = SinglePathDatasetSpeedUP(
        data_dir=cfg.val_dataset.path,
        transform=test_transforms,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=True,
        **_worker_kwargs(num_workers),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        **_worker_kwargs(max(0, num_workers // 2)),
    )
    return train_loader, val_loader
