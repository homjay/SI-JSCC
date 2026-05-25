# CBJSCC

PyTorch implementation for **Channel-Blind Joint Source-Channel Coding for
Wireless Image Transmission**.
The code is organized around the standard JSCC pipeline:

```text
input image -> encoder -> channel -> decoder -> reconstructed image
```

The repository is extracted from the earlier experimental codebase and keeps only
the model, channel, data loading, single-device training, and basic inference /
evaluation utilities needed for open-source use.

## Paper Code Notice

This repository is a cleaned and reorganized implementation of the code used for
our paper, **Channel-Blind Joint Source-Channel Coding for Wireless Image
Transmission**. It is intended for training, evaluation, and reproduction of the
main simulation behavior, rather than as a direct dump of the original
experimental workspace.

For the earlier [SI-JSCC](https://github.com/homjay/SI-JSCC) repository, the
`inference` branch keeps the original inference notebook and released
pretrained weights, while the `training` branch points to this cleaned
training/evaluation code. Pretrained checkpoints are not bundled in the
training code by default; users should either train models with the commands
below or use the `inference` branch when they need the original inference
assets.

Simulation runs with this reorganized code show the same overall behavior and
model ordering as the paper. Exact metric values can vary with training length,
optimizer settings, random initialization, and the selected checkpoint.

## Structure

```text
CBJSCC/
├── configs/              # Hydra configs for datasets, channels, and backbones
├── src/
│   ├── backbones/        # Encoder/decoder implementations
│   ├── channels/         # AWGN, dynamic AWGN, Rayleigh, identity channel helpers
│   ├── data/             # Image-folder dataset
│   ├── jscc.py           # JSCCAutoEncoder core pipeline
│   └── model_loader.py   # Backbone registry and model construction
├── train.py              # Single-GPU/CPU training entrypoint
├── evaluate.py           # Checkpoint evaluation CLI
├── predict_single_image.py
└── requirements.txt
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment if the default pip
resolver does not pick the right wheel.

## Dataset

By default the config expects image folders at:

```text
datasets/cifar10/train
datasets/cifar10/test
```

The loader recursively reads `.jpg`, `.jpeg`, and `.png` files. If an `index.txt`
exists in a dataset directory, each line is treated as a relative image path.

Override paths at runtime:

```bash
python train.py train_dataset.path=/path/to/train val_dataset.path=/path/to/val
```

## Train

Default training uses the `cbjscc` backbone, AWGN channel, one local device, and
TensorBoard logging:

```bash
python train.py
```

Useful overrides:

```bash
python train.py batch_size=64 test_batch_size=64
python train.py max_epochs=100 channel_type=dynamic_awgn
python train.py backbones@coder=adjscc
python train.py backbones@coder=deepjscc
```

Paper-aligned ImageNet training overrides:

```bash
export IMAGENET_TRAIN=/path/to/ILSVRC/Data/CLS-LOC/train
export KODAK_ROOT=/path/to/kodak

python train.py \
  train_dataset.path="${IMAGENET_TRAIN}" \
  val_dataset.path="${KODAK_ROOT}" \
  patch_size=128 batch_size=112 test_batch_size=24 \
  min_snr=-5 max_snr=20 learning_rate=0.0001 \
  loss_type=l1_charbonnier optimizer.type=lion 'optimizer.betas=[0.9,0.99]'
```

If ADJSCC fails on recent CUDA/PyTorch builds with a cuBLASLt Linear-layer
initialization error, run it with `DISABLE_ADDMM_CUDA_LT=1`.

This entrypoint intentionally does not use `torchrun`, `DistributedDataParallel`,
or distributed samplers.

## Checkpoints

Checkpoints are written under Hydra's run directory:

```text
checkpoints/YYYY-MM-DD/YYYY-MM-DD-HH-MM-SS_jscc-sensitivity-informed/checkpoints/
```

Resume with:

```bash
python train.py checkpoint=/path/to/checkpoint.pth
```

## Evaluate

```bash
python evaluate.py \
  --project checkpoints/YYYY-MM-DD/RUN_NAME \
  --dataset datasets/cifar10/test \
  --snrs 1,3,5,7,10,13
```

You can also pass explicit files:

```bash
python evaluate.py \
  --config checkpoints/RUN/configs/resolved_config.yaml \
  --checkpoint checkpoints/RUN/checkpoints/latest.pth \
  --dataset /path/to/images
```

For full-resolution Kodak evaluation, keep `--batch-size 1` unless you also use
`--crop`; the Kodak images include both landscape and portrait shapes.

## Single-Image Inference

```bash
python predict_single_image.py \
  --image input.png \
  --checkpoint checkpoints/RUN/checkpoints/latest.pth \
  --output reconstruction.png \
  --snr 10
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.
