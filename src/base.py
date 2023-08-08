#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/base.py
# Project: SI-JSCC
# Created Date: Monday, August 7th 2023, 9:17:22 pm
# Author: Shisui
# Copyright (c) 2023 Uchiha
# ----------	---	----------------------------------------------------------
###
# %%
import time
import numpy as np
import onnxruntime as ort
from skimage import metrics


class CoderSession:
    """Coder session for encoder and decoder."""

    def __init__(self, onnx_path="encoder.onnx", device="cpu", show_time=False) -> None:
        self.z_shape = (1, 16, 128, 192)
        self.show_time = show_time
        if device == "gpu":
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        self.coder_session = ort.InferenceSession(onnx_path, providers=providers)

    @staticmethod
    def signal2tensor(signal, shape):
        """convert signal to tensor."""
        z_out = np.concatenate(signal, axis=0).astype(np.float32)
        return np.reshape(z_out, shape)

    @staticmethod
    def complex_power_normalization(x: np.ndarray):
        """
        Power normalization.
        param x: input data, [batch, channel, height, width]
        """
        x = x.flatten()
        dim_z = x.shape[0] // 2
        z_in = np.array([x[:dim_z], x[dim_z:]])
        norm_factor = np.sum(z_in**2)
        return z_in * np.sqrt(dim_z / norm_factor)

    def encode(self, input_img) -> np.ndarray:
        if self.show_time:
            start = time.time()
        ort_inputs = {self.coder_session.get_inputs()[0].name: input_img}
        ort_outs = self.coder_session.run(None, ort_inputs)
        if self.show_time:
            print(f"Inference time: {time.time() - start:.4f}s")
        return self.complex_power_normalization(ort_outs[0])

    def decode(self, z) -> np.ndarray:
        if self.show_time:
            start = time.time()
        z_real = z.real
        z_imag = z.imag
        z = np.concatenate([z_real, z_imag], axis=0)
        z = self.signal2tensor(z, self.z_shape)
        ort_inputs = {self.coder_session.get_inputs()[0].name: z}
        ort_outs = self.coder_session.run(None, ort_inputs)
        if self.show_time:
            print(f"Inference time: {time.time() - start:.4f}s")
        return ort_outs[0]


def prepare_array_input(input_img):
    """parse input image to array for encoder.

    Args:
        input_img (np.darray): channel last image array

    Returns:
        np.darray: channel first image array with shape [1, 3, height, width]
    """
    input_img = np.transpose(input_img, (2, 0, 1))
    return np.expand_dims(input_img, axis=0)


def compare_psnr(x_pred, x):
    return metrics.peak_signal_noise_ratio(x_pred, x, data_range=1)


def reconstruct_image(decoded_data):
    """reconstruct image from decoded array.

    Args:
        decoded_data (array): decoded data

    Returns:
        _type_: image array type
    """
    reconstructed_img = np.transpose(decoded_data[0], (1, 2, 0))
    reconstructed_img = np.clip(reconstructed_img, 0, 1)
    return reconstructed_img
