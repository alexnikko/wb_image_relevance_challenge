# Cross-Modal Image-Text Relevance

A competition-ready PyTorch pipeline for ranking the relevance of marketplace images to product-card text. It combines a `timm` vision backbone with a Hugging Face text encoder, FiLM-style feature modulation, grouped validation, pairwise ranking loss, and EMA checkpoints.

## Architecture

```mermaid
flowchart LR
    I[Product image] --> V[timm vision encoder]
    T[Title + description] --> X[Transformer text encoder]
    X --> G[Sigmoid gate]
    V --> M[FiLM modulation]
    G --> M
    M --> C[Concatenate features]
    X --> C
    C --> H[Binary relevance head]
```

The text representation produces a gate over the image embedding. The modulated image features and text features are then concatenated and scored by a small MLP.

## Engineering highlights

- `GroupKFold` split by product card to prevent related samples leaking across train and validation.
- Staged fine-tuning: train the fusion head first, then unfreeze both encoders.
- BCE objective combined with an in-card BPR pairwise ranking loss.
- Exponential moving average weights for validation and checkpoint selection.
- Optional image preloading and per-card text embedding caches.
- Fold ensembling for final predictions.
- AMP-safe gradient clipping with a focused regression test and GitHub Actions check.

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs three commands: `make-folds`, `fusion-train`, and `fusion-infer`.

## Data

```text
data/
├── train.csv   # id, title, description, card_identifier_id, label
├── test.csv    # id, title, description, card_identifier_id
└── images/
    ├── 1.jpg
    └── ...
```

Each image filename must match its integer `id`. The binary `label` column is required only for training.

## Train

Create grouped folds:

```bash
make-folds --train_csv data/train.csv --n_splits 5
```

Run the fusion model:

```bash
fusion-train \
  --train_csv data/train_folds5.csv \
  --img_dir data/images \
  --out_dir outputs/fusion \
  --img_model eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \
  --txt_model deepvk/USER-bge-m3 \
  --image_size 448 \
  --epochs 8 \
  --batch_size 24 \
  --freeze_epochs 2 \
  --pair_lambda 0.01 \
  --cache_text \
  --grad_checkpoint
```

The best EMA checkpoint for each fold is written to `outputs/fusion/folds/fusion_fold{n}.pt`.

## Infer

```bash
fusion-infer \
  --test_csv data/test.csv \
  --img_dir data/images \
  --out_dir outputs/fusion \
  --img_model eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \
  --txt_model deepvk/USER-bge-m3 \
  --image_size 448 \
  --batch_size 64 \
  --cache_text \
  --preload_ram
```

Inference averages predictions across fold checkpoints and writes `outputs/fusion/sub/submission_fusion.csv` by default. Text embeddings are rebuilt after each fold checkpoint is loaded, so cached features always match that fold's encoder weights.

## Tests

```bash
python -m pip install pytest
pytest -q
```

The public repository does not include the competition dataset, trained checkpoints, or a leaderboard result. It contains the complete training and inference implementation needed to reproduce the pipeline with compatible data.

## License

[MIT](LICENSE)
