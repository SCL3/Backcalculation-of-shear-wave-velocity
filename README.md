# Backcalculation of Shear-Wave Velocity

Deep learning models that backcalculate a shear-wave velocity (Vs) profile
from a frequency / phase-velocity / amplitude (FVS) spectrum image, together
with the geometry of the receiver array used to record it (near offset `x0`,
receiver spacing `dx`, number of receivers `Ch`).

## Origin & credits

This project originates from the research of **Dr. Tran Quoc Kinh**, a PhD
candidate at the Civil Engineering laboratory of **National Yang Ming Chiao
Tung University (NYCU), Taiwan**. His baseline model (`ModelCNN_fvs` in
`model.py`) is the reference every experiment in this repository is compared
against.

The code was then extended by **Simon CHANTHRABOUTH-LIEBBE**, a Computer Science / Artificial
Intelligence student at **CY Tech, France**, during a 3-month internship
carried out at NYCU's Civil Engineering laboratory.

## Hardware & data requirements

- **GPU strongly recommended.** Training runs on a CUDA-capable NVIDIA GPU;
  falls back to Apple MPS or CPU otherwise (see `main.py`), but a full run on
  the 300K dataset takes roughly 1–2 days on GPU and is impractical on CPU.
- **VRAM**: the largest backbones (ResNet-50, DenseNet-121) train comfortably
  on an 8 GB GPU at the default batch size (64).
- **Disk space**: the datasets are made of individual `.mat` (MATLAB) files,
  one per sample (FVS image + geometry scalars + target Vs profile). The
  300K-sample dataset requires 45.5GB of free disk space; a 5K subset is
  also provided for quick sanity checks (`Model_100/5K_dataset_files`).
- **Dataset location**: the dataset itself is not included in this repository
  (private research data). Update the `data_folder` path in `main.py` /
  `Call_dataset.py` to point at your local copy before training.

## Installing dependencies

1. Install Python 3.10+ (a virtual environment is recommended).
2. Check your GPU driver's CUDA version:
   ```bash
   nvidia-smi
   ```
3. Get the exact PyTorch/CUDA install command for your setup from
   [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)
   and adjust the `cu126` index in `requirements.txt` if needed.
4. Install everything:
   ```bash
   pip install -r requirements.txt
   ```

## Repository layout

```
Model_6/                        6 layers model
Model_100/                      100 layers model
├── 5K_dataset_files/            Small dataset for fast sanity checks
└── 300K_dataset_files/
    ├── Call_dataset.py          Dataset loader (.mat files -> tensors)
    ├── model.py                 Baseline + "fvs" model family
    ├── model_test_geo.py        Latest, physics-informed "geo" model family
    ├── train.py / test.py       Training and evaluation loops
    ├── main.py                  Entry point: pick models, run training/testing
    ├── losses.py / loss_fcns.py Loss functions (RMSE, etc.)
    └── PTH/, log/, runs/, figures/, validation_results/   Checkpoints & outputs
```

Run training/testing from inside `Model_100/300K_dataset_files/`:

```bash
python main.py --mode train   # or --mode test
```

## Model families

All models take an FVS spectrum image plus a handful of geometry scalars
(`x0`, `dx`, `Ch`) and output a 101-point Vs(depth) profile. They differ in
**how the image is processed** and **how the geometry is fed into the
network**.

### Baseline (Dr. Kinh) — `ModelCNN_fvs`
A pretrained **ResNet-50** processes the FVS image. The near-offset `x0` is
handled by literally repeating that single scalar across a 76×191 grid and
feeding it through a 30M-parameter `Flatten -> Linear` branch. As shown by the
analysis in `model_test_geo.py`, repeating one value like this is
mathematically equivalent to a 1-parameter linear function, the 30M
parameters buy no expressiveness and, worse, distort the effective learning
rate on that branch by several orders of magnitude.

### Backbone family
Swapping the baseline's image backbone while keeping the same (flawed)
geometry branch, to compare architectures on the FVS image itself:
ResNet-34/50/101, DenseNet-121, Swin-T, EfficientNet-B0, and a from-scratch
custom CNN (`ModelCustomCNN_fvs`, no ImageNet pretraining).

### "fvs" branch of work - `model.py`, `v2` variants
The first redesign target: replacing the baseline's oversized, ill-posed
geometry branch with a proper small MLP that encodes the scalar geometry
directly (`ModelCNN_fvs_v2`), then progressively adding:
- **FiLM conditioning** (`_v2_film`): the geometry modulates the image
  features multiplicatively instead of only being concatenated to them.
- **All 4 geometric scalars** (`_v2_all_geo`, `_v2_all_geo_film`): `x0`, `dx`,
  `Ch`, and derived array length `L = (Ch-1)*dx`, instead of `x0` alone.

### "geo" branch of work - `model_test_geo.py` (latest, best-performing models)
The current, most physics-aware iteration of the geometry branch (config
**G3**), applied on top of three image backbones:

| Model | Image backbone | Backbone output |
|---|---|---|
| `ModelCNN_fvs_v3_geo` | ResNet-50 | 1000-d |
| `ModelDenseNet121_fvs_geo` | DenseNet-121 | 1024-d |
| `ModelEfficientNetB0_fvs_geo` | EfficientNet-B0 | 1280-d |

What makes this family different:

- **`GeometryFeatures`** - a zero-parameter feature engineering step that
  turns `(x0, dx, Ch)` into physically meaningful, normalized quantities
  (aperture, near-field/aliasing slopes, `x0` in metres, one-hot encoding of
  the discrete `dx`/`Ch` values) instead of feeding raw scalars to the
  network.
- **Fourier/sine-cosine encoding** of those features (Tancik et al., 2020),
  which removes the spectral bias that makes small MLPs struggle to fit
  sharp functions of low-dimensional inputs.
- **Low-rank FiLM** (`LowRankFiLM`) - the geometry multiplicatively modulates
  the image features (`(1+γ)·f + β`) through a rank-16 bottleneck, so the
  interaction is expressive without exploding the parameter count. It is
  zero-initialized so training starts identical to plain concatenation and
  the modulation is learned progressively.
- **Radically fewer parameters, correctly conditioned**: the full geometry
  branch (G3: physics features + Fourier encoding + low-rank FiLM) uses
  ~49k parameters, vs. the baseline's 30.8M, about **625× smaller**, while
  fixing the gradient/learning-rate pathology described above. Two cheaper
  ablations (G1: physics features only, G2: + Fourier encoding, no FiLM) are
  also implemented to isolate the contribution of each idea.

Running the file directly executes a self-test (numerical proof of the
baseline's redundancy, parameter counts per variant, and shape checks):

```bash
python model_test_geo.py
```
