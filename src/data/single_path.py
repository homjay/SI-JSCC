#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2026 Jay
# Licensed under the MIT License. See LICENSE in the project root.

import os
import torch
import numpy as np
import random
import math
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%


def gen_noise(
    shape=None,
    channel_type="awgn",  # rayleigh
    eval_snr=None,
    min_snr=1,
    max_snr=13,
    snr_list=None,
):
    if shape is None:
        shape = 8192
    # with torch.no_grad():
    if snr_list is not None:
        snr = random.choice(snr_list)
        snr = torch.tensor(snr)
    elif eval_snr is None:
        snr = torch.rand(1) * (max_snr - min_snr) + min_snr
    elif isinstance(eval_snr, torch.Tensor):
        snr = eval_snr
    else:
        snr = torch.tensor(eval_snr)
    noise_stddev = torch.sqrt(10 ** (-snr / 10))
    # noise_stddev = torch.complex(noise_stddev, torch.tensor(0.0))
    noise = torch.normal(mean=0, std=1 / math.sqrt(2), size=[shape, 2])
    # noise_imag = torch.normal(mean=0, std=1 / math.sqrt(2), size=shape)
    # noise = torch.(noise_real, noise_imag)
    if channel_type == "rayleigh":
        h = torch.sqrt(
            torch.normal(mean=0.0, std=1, size=(shape,)) ** 2
            + torch.normal(mean=0.0, std=1, size=(shape,)) ** 2
        ) / np.sqrt(2)
    elif channel_type == "awgn":
        h = torch.ones(shape)
    else:
        raise ValueError(f"channel type:{channel_type}, must be 'awgn' or 'rayleigh'")
    return h, noise_stddev * noise, snr


def gen_awgn_noise(shape=None, eval_snr=None, min_snr=1, max_snr=13):
    if shape is None:
        shape = 6144
    with torch.no_grad():
        if eval_snr is None:
            snr = torch.rand(1) * (max_snr - min_snr) + min_snr
        elif isinstance(eval_snr, torch.Tensor):
            snr = eval_snr
        else:
            snr = torch.tensor(eval_snr)
        noise_stddev = torch.sqrt(10 ** (-snr / 10))
        noise_stddev = torch.complex(noise_stddev, torch.tensor(0.0))
        noise = torch.complex(
            torch.normal(
                mean=0,
                std=1 / math.sqrt(2),
                size=(shape,),
            ),
            torch.normal(
                mean=0,
                std=1 / math.sqrt(2),
                size=(shape,),
            ),
        )
    return noise_stddev * noise, snr


class SinglePathDatasetSpeedUP(Dataset):
    """compress image dataset"""

    def __init__(
        self,
        data_dir,
        transform=None,
    ):
        self.img_collection = []
        self.path_gan_collection = []
        self.transform = transform
        self.index_images(data_dir)

    def __len__(self):
        return len(self.img_collection)

    def index_images(self, img_dir):
        """index images in the given directory
        Args:
            img_dir (str): image directory
        """
        index_file = os.path.join(img_dir, "index.txt")
        if os.path.isfile(index_file):
            with open(index_file, "r") as f:
                lines = f.readlines()
            self.img_collection = [line.strip() for line in lines]
            self.img_collection = [
                os.path.join(img_dir, img) for img in self.img_collection
            ]
        else:
            for root, _, files in os.walk(img_dir):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.img_collection.append(os.path.join(root, fname))

    def __getitem__(self, idx):
        img_path = self.img_collection[idx]
        # image = read_image(img_path,  ImageReadMode.RGB)
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        # if self.path_gan_dir is None:
        return image


def get_transforms(patch_size=128, crop=True):
    if crop:
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(patch_size, pad_if_needed=True, fill=0),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        test_transforms = transforms.Compose(
            [
                transforms.CenterCrop(patch_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    return train_transforms, test_transforms
