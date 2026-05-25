#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for CBJSCC training.
"""

import logging
import warnings


def setup_training_environment():
    """Set quiet defaults for common third-party warnings."""

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    logging.getLogger("torch").setLevel(logging.WARNING)


def configure_pytorch():
    """Configure PyTorch performance settings when the runtime supports them."""
    import torch

    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = True

    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = True


def print_system_info():
    """Print system information for debugging."""
    import platform

    import torch

    print("=" * 50)
    print("CBJSCC Training Environment")
    print("=" * 50)
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPUs: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"   GPU {index}: {torch.cuda.get_device_name(index)}")
    else:
        print("CPU-only mode")

    print("=" * 50)
