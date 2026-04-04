# ANRF-AISE-Hack-2026-Flood-Detection

<div align="center">

![AISEHack](https://img.shields.io/badge/AISEHack-Theme%201-blue?style=flat-square)
![Model](https://img.shields.io/badge/Model-Prithvi--EO--2.0--300M--TL-green?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20Lightning-purple?style=flat-square)
![Submission](https://img.shields.io/badge/Mid%20Submission-Stage%202-orange?style=flat-square)

Transfer Learning from a Flood-Finetuned Geospatial Foundation Model

</div>

---

## Overview

This notebook fine-tunes **Prithvi-EO-2.0-300M-TL-Sen1Floods11** — an IBM × NASA geospatial foundation model already trained for flood segmentation on the Sen1Floods11 benchmark — for a custom dataset combining SAR (HH, HV) and optical (Green, Red, NIR, SWIR) bands.

Unlike backbone-only transfer, Stage 2 loads the **full checkpoint**: ViT-Large encoder, UperNet decoder, and segmentation head. The model starts with genuine flood-detection knowledge, requiring less training to converge.

**HuggingFace model:** [`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)

---

## Stage 2 vs Stage 1

| | Stage 1 | Stage 2 |
|---|---|---|
| Backbone | ViT-Huge (600M, 1280-dim, 32 layers) | ViT-Large (300M, 1024-dim, 24 layers) |
| Decoder init | Random | Pretrained on Sen1Floods11 |
| Head init | Random | Pretrained on Sen1Floods11 |
| Batch size | 2 | 4 |
| Gradient accumulation | 8 | 4 |
| Decoder LR multiplier | 10× | 5× |
| Expected convergence | Slower | Faster |

---

## Model Architecture

```
Input — 6 bands × 224 × 224
        │
        ▼
  ViT-Large Encoder
  embed_dim=1024 | depth=24 | patch_size=16
  Feature hooks at layers: [5, 11, 17, 23]
        │
        ▼
  UperNet Decoder
  channels=256 | PSP (pool_scales=1,2,3,6) | FPN | scale_modules=True
        │
        ▼
  Segmentation Head
  Dropout(0.1) → Conv2d(256 → 2) → Bilinear upsample → 512×512
        │
        ▼
  Output — binary mask (0=background, 1=flood)
```

**Checkpoint structure** (`Prithvi-EO-V2-300M-TL-Sen1Floods11.pt`, 1.28 GB):

```
model.encoder.*   →  ViT-Large backbone
model.decoder.*   →  UperNet (PSP + FPN + scale modules)
model.head.*      →  SegmentationHead classifier
```

---

## Band Configuration

| Slot | Pretrained Band | Our Band | Transfer |
|---|---|---|---|
| 0 | BLUE | — | Not used |
| 1 | GREEN | Green | ✅ Pretrained filter copied |
| 2 | RED | Red | ✅ Pretrained filter copied |
| 3 | NIR_NARROW | NIR | ✅ Pretrained filter copied |
| 4 | SWIR_1 | SWIR | ✅ Pretrained filter copied |
| — | (new) | HH (SAR) | 🆕 Randomly initialized |
| — | (new) | HV (SAR) | 🆕 Randomly initialized |

4 of 6 pretrained optical filters are reused. SAR channels (HH, HV) are newly initialized in the patch embedding, allowing the model to learn SAR representations while retaining its optical flood-detection knowledge.

---

## Transfer Strategy

**Smart Patch Embed Transfer** maps band names to pretrained filter indices:

```python
BAND_TO_PRETRAINED_INDEX = {
    'Green': 1,  # ← pretrained GREEN filter
    'Red':   2,  # ← pretrained RED filter
    'NIR':   3,  # ← pretrained NIR_NARROW filter
    'SWIR':  4,  # ← pretrained SWIR_1 filter
}
# HH, HV → random init
```

**Transfer mode** is configurable via `TRANSFER_MODE`:

```python
TRANSFER_MODE = 'full'     # encoder + decoder + head (default)
TRANSFER_MODE = 'encoder'  # encoder only, decoder random
TRANSFER_MODE = 'frozen'   # encoder loaded and frozen
```

**Differential learning rates:**

| Component | Multiplier | Reason |
|---|---|---|
| Encoder (ViT) | 1× (3e-5) | Gentle fine-tuning |
| Decoder (UperNet) | 5× | Pretrained, needs moderate adjustment |
| Head (classifier) | 10× | Highest plasticity |

---

## Channel Engineering

SAR data is preprocessed before training:

**1. Refined Lee Speckle Filter** (HH, HV only)
Applies an 8-direction MMSE filter to suppress speckle while preserving edges. For each pixel, the most homogeneous directional sub-window is selected (lowest coefficient of variation), and the MMSE weight is computed as:

```
w = clamp((local_var − noise_var) / local_var, 0, 1)
noise_var = mean² / n_looks
result = mean + w × (pixel − mean)
```

**2. dB Conversion** (HH, HV only)
```python
db = 10 * log10(max(value, 1e-10))
```
Compresses SAR dynamic range for better gradient behavior during training.

**3. Channel Cache**
Preprocessed arrays are saved as `.npy` files to avoid recomputing the Refined Lee filter each epoch.

**4. Augmentation** (training only)
Horizontal flip, vertical flip, random 90° rotation — all at `p=0.5`.

---

## Training Configuration

```python
EPOCHS          = 60
BATCH_SIZE      = 4           # effective batch = 16
ACCUM_GRAD      = 4
LR              = 3e-5
WEIGHT_DECAY    = 0.05
HEAD_DROPOUT    = 0.1
CLASS_WEIGHTS   = [0.25, 0.75]  # flood pixels are 3–9× rarer
IMG_SIZE        = 224
WARMUP_ITERS    = 15
WARMUP_RATIO    = 1e-6
ES_PATIENCE     = 35
AUX_LOSS_WEIGHT = 0.4
SEED            = 42
```

**LR Schedule:** Linear warmup for 15 iterations → CosineAnnealing to 0.

**Early stopping:** Monitors `val/mIoU`, stops after 35 epochs without improvement.

**Metrics tracked:** `val/mIoU`, `val/Overall_Accuracy`, `val/boundary_mIoU`, `test/mIoU`.

---

## Inference & Submission

Full-resolution (512×512) prediction uses sliding window inference since the model is trained at 224×224:

```
window = 224  |  stride = 112  (50% overlap)
Logits averaged over overlapping windows → argmax → final mask
```

RLE encoding uses **column-major (Fortran) order** to match the competition format:

```python
pixels = mask.flatten(order='F')   # critical — not row-major
```

Submission pipeline: best checkpoint → sliding window inference → RLE per image → `submission.csv`.

---

## Project Structure

```
experiments/
└── {exp_name}/
    ├── norm_stats.json          # per-channel mean/std
    ├── results.json             # metrics + config
    ├── best_model.ckpt          # best checkpoint (val/mIoU)
    ├── submission_{name}.csv    # Kaggle submission
    └── channel_cache/           # preprocessed .npy arrays
```

---

## Setup & Usage

**Install dependencies:**
```bash
pip install lightning einops huggingface_hub rasterio scipy timm albumentations torchmetrics
```

**Key toggles before running:**
```python
LOAD_SEN1FLOODS11 = True    # download and use pretrained weights
TRANSFER_MODE     = 'full'  # full | encoder | frozen
FREEZE_BACKBONE   = False
USE_TPU           = False   # set True on Kaggle TPU
```

Run all cells top to bottom. The notebook handles dependency install → config → data loading → channel caching → normalization → training → prediction visualization → submission generation.

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch` / `torchvision` | Core deep learning |
| `lightning` | Training loop & checkpointing |
| `timm` | ViT-Large backbone |
| `rasterio` | GeoTIFF I/O |
| `albumentations` | Augmentation |
| `torchmetrics` | mIoU & accuracy |
| `huggingface_hub` | Pretrained weight download |
| `scipy` | Directional convolutions (Refined Lee) |
| `einops` | Tensor rearrangement |
