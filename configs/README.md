# Configuration

CBJSCC uses Hydra YAML configs. The main entrypoint is `configs/main.yaml`.

Switch the model backbone with Hydra defaults:

```bash
python train.py backbones@coder=cbjscc
python train.py backbones@coder=adjscc
python train.py backbones@coder=deepjscc
```

Override common training values directly on the command line:

```bash
python train.py \
  train_dataset.path=/path/to/train \
  val_dataset.path=/path/to/val \
  batch_size=64 \
  channel_type=awgn
```

Backbone configs live in `configs/backbones/`; dataset path configs live in
`configs/datasets/`.
