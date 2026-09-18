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
  Mixed precision (AMP, fp16) is enabled automatically on CUDA.
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
    ├── Call_dataset.py          Dataset loader (.mat files -> tensors) + augmentation
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

## Quick start

Everything you normally touch lives in the configuration block at the top of
`main.py`:

```python
IN_INSTANCES = ['fvs', 'x0', 'dx', 'Ch']
IN_CHANNELS  = 3        # channels READ from the .mat image (1 = amplitude only, 3 = all)

ADD_NOISE           = [("gaussian", 0.02), ("mask", 2, 15.0, 60.0)]  # training augmentation
VAL_ADD_NOISE       = None                                           # validation stays clean
RETURN_MASK_CHANNEL = True                                           # feed the mask to the model

TRAIN_RATIO = 0.8 ;  NUM_EPOCHS = 250 ;  EARLY_STOPPING = 25
BATCH_SIZE  = 64  ;  NUM_WORKERS = 4
```

Then pick the models to run in the `models = [...]` list inside `run_train()`
and launch `python main.py --mode train`.

Before touching anything, check that the loader sees your data correctly:

```bash
python Call_dataset.py      # prints the 4 augmentation configs and their dataset sizes
```

(Edit the `input_dir` / `output_dir` paths in that file's `__main__` block first.)

## Data layout

Each `.mat` file holds a struct whose first field (`fvs`) is a **3 × 76 × 191**
array plus the geometry scalars:

| Channel | Content | Range |
|---|---|---|
| 0 | normalized frequency `f' = f·L / Vs_max` | `f` : 5 → 80 Hz |
| 1 | normalized phase velocity `Vs' = Vs / Vs_max` | `Vs` : 50 → 1000 m/s |
| 2 | **amplitude**, the only real measurement | 0 → 1 |

- axis 1 (**76 rows**) = frequency, 5 Hz → 80 Hz, 1 Hz step
- axis 2 (**191 columns**) = phase velocity, 50 → 1000 m/s, 5 m/s step

The amplitude is normalized per frequency row at load time
(`amplitude / amplitude.max(axis=1)`), so **every visible row has a maximum of
exactly 1.0**. The target is a 101-point Vs(depth) profile read from the
matching CSV in `output100/`.

## Data augmentation (`Call_dataset.py`)

Augmentation is driven by a single argument, `add_noise`. It is either `None`
or a list holding either or both of two specs. `ADD_NOISE` in `main.py` is
passed straight through to the dataset.

```python
ADD_NOISE = None                                            # clean,      dataset ×1
ADD_NOISE = [("gaussian", 0.02)]                            # noise only, dataset ×1
ADD_NOISE = [("mask", 2, 15.0, 60.0)]                       # mask only,  dataset ×3
ADD_NOISE = [("gaussian", 0.02), ("mask", 2, 15.0, 60.0)]   # both,       dataset ×3
```

### `("gaussian", noise_std)`

Additive Gaussian noise on the amplitude channel, in the units of the
normalized amplitude. Drawn from the global RNG, so a **new realisation at
every epoch**. The dataset size is unchanged.

### `("mask", mask_number, mask_min, mask_max)`

Simulates a **band-limited field record**: with a sledgehammer source the low
frequencies carry little energy and the high frequencies are attenuated or
aliased, so a real dispersion image is only usable over part of the 5-80 Hz
axis. For every masked variant, two cut-off frequencies are drawn:

```
f_low  ~ U(5, mask_min)      rows with f <= f_low   are hidden
f_high ~ U(mask_max, 80)     rows with f >= f_high  are hidden
```

The network only sees the open band `(f_low, f_high)`. Because
`f_low <= mask_min` and `f_high >= mask_max` by construction, the **core band
`(mask_min, mask_max)` always survives**, whatever the draw.

`mask_number` **multiplies the dataset**: each `.mat` file yields
`mask_number` masked variants, each with its own independent pair of cut-offs,
plus one pristine variant when `INCLUDE_CLEAN` is on.

```
len(dataset) = n_files × (mask_number + 1 if INCLUDE_CLEAN else mask_number)
```

So `("mask", 2, 15.0, 60.0)` with the default settings gives **3 samples per
file** and epochs take **3× longer**. Lower `MAX_FILES` in `Call_dataset.py` to
compensate.

### Why hidden cells hold `-1.0` and not `0.0`

Writing `0.0` into the hidden rows would tell the network *"the amplitude is
zero at this frequency"*, a real, informative measurement, instead of
*"this value is hidden, infer it"*. The two are not distinguishable locally:
about **45 % of the 3×3 patches of a real dispersion image are already
all-dark**, so a zero-filled patch looks exactly like a genuine low-energy
patch. Telling them apart would require computing "is the maximum over the
whole 191-px row equal to zero?", which no early convolution layer can do.

`MASK_FILL_VALUE = -1.0` sits **outside the valid amplitude range `[0, 1]`**, so
a single pixel is already unambiguous in the very first conv layer.

### Module constants (top of `Call_dataset.py`)

| Constant | Default | Meaning |
|---|---|---|
| `F_MIN_HZ`, `F_MAX_HZ` | `5.0`, `80.0` | physical frequency axis of the image |
| `MAX_FILES` | `300000` | number of `.mat` **files** used (the mask multiplies this into samples) |
| `MASK_FILL_VALUE` | `-1.0` | value written into hidden cells; must be outside `[0, 1]` |
| `INCLUDE_CLEAN` | `True` | keep one pristine variant per file (no noise **and** no mask) |
| `MASK_SEED` | `1234` | base entropy for the masks |

**`INCLUDE_CLEAN = True` matters.** Without it, not a single clean full-band
image is ever shown to the model during training, and its accuracy on clean
synthetic data drops. Variant 0 of every file bypasses *both* augmentations.

**Masks are deterministic.** Variant *v* of file *i* always gets the same
cut-offs, seeded from the sample index, independently of the global RNG. The
expanded dataset is therefore a fixed, reproducible dataset: identical across
epochs, across DataLoader workers and across machines. `ds.mask_bounds(idx)`
returns the `(f_low, f_high)` of any sample without loading its `.mat` file.

### `RETURN_MASK_CHANNEL`

Appends a fourth, binary channel to the FVS image: `1` = measured, `0` =
hidden. The models are then built with `IN_CHANNELS + 1` channels - `main.py`
handles this through `MODEL_IN_CHANNELS`, and all three "geo" models rebuild
their first convolution from `in_channels`, so no model code changes.

> **This is not an "ignore these pixels" flag.** The loss is computed on the Vs
> profile only and is never masked, and the target is the **full** 101-point
> profile for every variant - a sample whose 5–13 Hz band was hidden is graded
> on exactly the same deep layers as the pristine one. The channel only tells
> the network *where* data is missing, so that it knows it is guessing instead
> of believing the amplitude is zero there.
>
> Existing `.pth` checkpoints were trained with 3 channels. Set
> `RETURN_MASK_CHANNEL = False` to reload them.

### Validation

Validation is driven by `VAL_ADD_NOISE`, independently of `ADD_NOISE`. Keep it
at `None` (the default) so the val loss stays clean, deterministic and
comparable across runs. Set it to the same masking spec only if you
deliberately want the val loss to measure robustness to band-limited data,
the masks being deterministic, it stays reproducible either way.

### Common configurations

| Goal | `ADD_NOISE` | `INCLUDE_CLEAN` | Dataset size |
|---|---|---|---|
| Reference run | `None` | - | ×1 |
| Noise robustness | `[("gaussian", 0.02)]` | - | ×1 |
| One mask per sample, same epoch cost | `[("mask", 1, 15.0, 60.0)]` | `False` | ×1 |
| Band robustness, keeps clean data | `[("mask", 2, 15.0, 60.0)]` | `True` | ×3 |
| Both | `[("gaussian", 0.02), ("mask", 2, 15.0, 60.0)]` | `True` | ×3 |

## Training loop (`train.py`)

`train_model()` runs one full training + validation loop for a single model.
Beyond the augmentation arguments above:

- **Optimizer**: AdamW (`lr = 3e-4`, `weight_decay = 1e-4` - set explicitly,
  since AdamW defaults to `1e-2`), gradient-norm clipping at 5.0.
- **LR schedule**: 5 epochs of linear warmup, then cosine decay clamped flat at
  `ETA_MIN` past `COSINE_HORIZON = 120`. Plain `CosineAnnealingLR` is periodic
  (period `2·T_max`), which is why a `LambdaLR` is used instead.
- **Model selection / early stopping** run on the **pooled RMSE**
  (`sqrt(SSE / n_elements)`), the only variant that does not shift when
  `batch_size` changes. An improvement must beat the best by more than
  `MIN_DELTA_REL` (0.1 %) to reset patience.
- **Resume**: `resume=True` reloads `PTH/last_<model_name>.pth` (model,
  optimizer, scheduler, AMP scaler, epoch, best error, patience) and keeps
  logging into the same run. The best weights are saved separately as
  `PTH/saved_best_model_<run_id>.pth`.
- **Logging**: one CSV row per epoch in `log_path`, plus TensorBoard events in
  `log_dir/<run_id>`.

### The train/val split is done **by file**, not by sample

This is critical once masking is enabled: a file produces several variants that
all share the **same Vs profile** (the same output CSV). Splitting on sample
indices would scatter those variants across train and validation, so the model
would be validated on profiles it has already been trained on - an
optimistically low val loss and a broken early-stopping criterion. `train.py`
therefore permutes **file** indices, then expands them with
`dataset.indices_for_files(...)`. With no mask (1 variant per file) this is
strictly equivalent to a plain index split.

`Call_dataset.py` also sorts the `.mat` files numerically rather than relying on
`os.listdir` order, which is arbitrary: the deterministic masks and the split are
both indexed on that list, so a stable order is what makes a run reproducible.

## Model families

All models take an FVS spectrum image plus a handful of geometry scalars
(`x0`, `dx`, `Ch`) and output a 101-point Vs(depth) profile. They differ in
**how the image is processed** and **how the geometry is fed into the
network**.

### Baseline (Dr. Kinh) - `ModelCNN_fvs`
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

All three "geo" models rebuild their first convolution from `in_channels`, so
they accept the 4-channel input produced by `RETURN_MASK_CHANNEL = True`
without any modification.

Running the file directly executes a self-test (numerical proof of the
baseline's redundancy, parameter counts per variant, and shape checks):

```bash
python model_test_geo.py
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `size mismatch` on `load_state_dict` | The checkpoint was trained with a different channel count. Match `RETURN_MASK_CHANNEL` to the run that produced it, and delete `PTH/last_<model_name>.pth` before starting a new configuration (`resume=True` would reload stale weights and a `best_error` computed on another split). |
| Epochs suddenly 3× longer | Expected with `("mask", 2, ...)`: the dataset is 3× larger. Lower `MAX_FILES` in `Call_dataset.py`. |
| `NaN` in the loss | A raw amplitude row that is entirely zero makes the per-frequency normalization divide by zero. Check with `(fvs[-1].max(axis=1) == 0).any()` over the dataset. |
| Val loss far below train loss | With masking enabled, verify the split is still done by file - training samples are degraded while validation ones are clean, so a gap is normal, but a *large* one suggests leakage. |
| `ValueError: Need 5.0 <= mask_min < mask_max <= 80.0` | The mask band must lie inside the physical frequency axis. |
