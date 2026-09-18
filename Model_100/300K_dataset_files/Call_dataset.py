"""
Dataset loader for the MASW / FVS deep-learning inversion.

Data layout
-----------
Each .mat file stores a struct whose first field ('fvs') is a 3 x 76 x 191 array:
    channel 0 : normalized frequency        f'  = f * L / Vs_max     (f  : 5 -> 80 Hz)
    channel 1 : normalized phase velocity   Vs' = Vs / Vs_max        (Vs : 50 -> 1000 m/s)
    channel 2 : normalized amplitude in [0, 1]   <- the only real measurement

    axis 1 (76 rows)     = frequency      (5 Hz -> 80 Hz, 1 Hz step)
    axis 2 (191 columns) = phase velocity (50 m/s -> 1000 m/s, 5 m/s step)

Augmentation
------------
One argument, `add_noise`. None, or a list holding either or both specs:

    ("gaussian", noise_std)
        Additive noise on the amplitude, redrawn at every epoch.
        Does NOT change the dataset size.

    ("mask", mask_number, mask_min, mask_max)
        Band-limiting mask. Two cut-off frequencies are drawn per masked variant:
            f_low  ~ U(5, mask_min)      rows with f <= f_low  are hidden
            f_high ~ U(mask_max, 80)     rows with f >= f_high are hidden
        The network only sees the open band (f_low, f_high); the core band
        (mask_min, mask_max) always survives.
        MULTIPLIES THE DATASET: each .mat file yields `mask_number` masked variants
        (plus one pristine variant when INCLUDE_CLEAN is True).

Examples
--------
    add_noise = None                                          # clean
    add_noise = [("gaussian", 0.02)]                          # noise only
    add_noise = [("mask", 2, 15.0, 60.0)]                     # mask only
    add_noise = [("gaussian", 0.02), ("mask", 2, 15.0, 60.0)] # both
"""

import os

import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader

# --- Physical frequency axis of the FVS image (76 rows, 1 Hz step) ---
F_MIN_HZ = 5.0
F_MAX_HZ = 80.0

# --- Number of .mat FILES used. The mask multiplies this into samples. ---
MAX_FILES = 300000

# --- Value written into hidden cells. MUST be outside the valid amplitude range
# [0, 1], so that a single pixel already means "hidden, infer me" instead of
# "amplitude = 0 at this frequency". With 0.0 it would be locally identical to a
# genuine low-energy cell (~45% of a real dispersion image is already near zero). ---
MASK_FILL_VALUE = -1.0

# --- Keep one pristine variant per file (no noise AND no mask), so the model still
# sees clean full-band images during training. Only applies when a mask is used. ---
INCLUDE_CLEAN = True

# --- Base entropy for the masks. Variant v of file i always gets the same cut-offs,
# so the expanded dataset is fixed and reproducible across epochs and workers. ---
MASK_SEED = 1234


class MyDataset(Dataset):
    def __init__(self, in_instances, in_channels, input_dir, output_dir=None,
                 add_noise=None, return_mask_channel=False):
        """
        add_noise : None, or a list of ("gaussian", std) / ("mask", n, f_min, f_max).
        return_mask_channel : append a binary channel (1 = measured, 0 = hidden) to the
            FVS image. The model must then be built with in_channels + 1 channels.
            !!! This is NOT an "ignore these pixels" flag: the loss is never masked and
            the target stays the FULL Vs profile for every variant. It only tells the
            network WHERE the data is missing, so it knows that it is guessing.
        """
        # Sorted numerically ("0.mat", "1.mat", ..., "10.mat"), NOT in os.listdir order:
        # the masks and the train/val split are indexed on this list, so a stable order
        # is what makes a run reproducible.
        def _key(name):
            stem = os.path.splitext(name)[0]
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        self.input_files = [os.path.join(input_dir, f)
                            for f in sorted((f for f in os.listdir(input_dir)
                                             if f.endswith('.mat')), key=_key)]
        self.input = input_dir
        self.in_instances = in_instances
        self.in_channels = in_channels
        self.return_mask_channel = return_mask_channel

        self.noise_std, self.mask = self._parse(add_noise)
        self.n_base_files = min(MAX_FILES, len(self.input_files))
        self.n_variants = 1 if self.mask is None else \
            self.mask[0] + (1 if INCLUDE_CLEAN else 0)
        self.desc = self._describe()

        if output_dir is not None:
            self.output_dir = output_dir
            self.output_files = []
            for main_file in self.input_files:
                file_index = os.path.splitext(os.path.basename(main_file))[0]
                self.output_files.append(os.path.join(output_dir, f"{file_index}.csv"))
        else:
            self.output_dir = None
            self.output_files = None

    # -------------------------------------------------------------------
    @staticmethod
    def _parse(add_noise):
        """Turn the `add_noise` list into (noise_std, (mask_number, mask_min, mask_max))."""
        noise_std, mask = None, None
        for spec in (add_noise or []):
            kind = str(spec[0]).lower()
            if kind == "gaussian":
                noise_std = float(spec[1])
                if noise_std <= 0:
                    raise ValueError("noise_std must be > 0")
            elif kind == "mask":
                n, lo, hi = int(spec[1]), float(spec[2]), float(spec[3])
                if n < 1:
                    raise ValueError("mask_number must be >= 1")
                if not (F_MIN_HZ <= lo < hi <= F_MAX_HZ):
                    raise ValueError(f"Need {F_MIN_HZ} <= mask_min < mask_max <= {F_MAX_HZ}, "
                                     f"got {lo} and {hi}")
                mask = (n, lo, hi)
            else:
                raise ValueError(f"Unknown augmentation {spec[0]!r}")
        return noise_std, mask

    def _describe(self):
        parts = []
        if self.noise_std is not None:
            parts.append(f"gaussian(std={self.noise_std})")
        if self.mask is not None:
            n, lo, hi = self.mask
            parts.append(f"mask(n={n}, keep {lo}-{hi}Hz, fill={MASK_FILL_VALUE}"
                         f"{', +pristine' if INCLUDE_CLEAN else ''}) -> x{self.n_variants}")
        return " + ".join(parts) if parts else "none"

    def indices_for_files(self, file_indices):
        """Expand FILE indices into the sample indices they own. Used by train.py to
        split train/val by file, so two variants of the same .mat can never land on
        both sides of the split (they share the same Vs profile)."""
        return [i * self.n_variants + v
                for i in file_indices for v in range(self.n_variants)]

    def mask_bounds(self, idx):
        """The (f_low, f_high) cut-offs of sample `idx`, without loading the .mat file.
        (None, None) when the sample is not masked."""
        if self.mask is None:
            return None, None
        if INCLUDE_CLEAN and idx % self.n_variants == 0:
            return None, None
        _, lo, hi = self.mask
        rng = np.random.default_rng([MASK_SEED, idx])
        return rng.uniform(F_MIN_HZ, lo), rng.uniform(hi, F_MAX_HZ)

    # -------------------------------------------------------------------
    def _add_gaussian_noise(self, amplitude):
        """Global RNG -> a new realisation at every epoch."""
        noise = np.random.normal(0.0, self.noise_std,
                                 size=amplitude.shape).astype(amplitude.dtype)
        return np.clip(amplitude + noise, 0.0, None)

    def _apply_mask(self, amplitude, ascending, idx):
        """Hide the low- and high-frequency ends. Bounds included, as asked.
        Called AFTER the per-frequency normalization (otherwise max(axis=1) would be 0
        on hidden rows -> NaN) and AFTER the noise (so hidden cells hold the exact
        sentinel value). Returns (amplitude, hidden_rows)."""
        _, lo, hi = self.mask
        rng = np.random.default_rng([MASK_SEED, idx])
        f_low = rng.uniform(F_MIN_HZ, lo)
        f_high = rng.uniform(hi, F_MAX_HZ)

        row_f = np.linspace(F_MIN_HZ, F_MAX_HZ, amplitude.shape[0])
        if not ascending:
            row_f = row_f[::-1]
        hidden = (row_f <= f_low) | (row_f >= f_high)
        amplitude[hidden, :] = MASK_FILL_VALUE
        return amplitude, hidden

    # -------------------------------------------------------------------
    def __len__(self):
        # n_files * variants. The mask DOES grow the dataset: ("mask", 2, ...) with
        # INCLUDE_CLEAN gives 3 samples per file -> epochs 3x longer.
        return self.n_base_files * self.n_variants

    def __getitem__(self, idx):
        file_idx, variant = divmod(idx, self.n_variants)
        # Variant 0 is the pristine copy: no noise, no mask.
        is_clean = self.mask is not None and INCLUDE_CLEAN and variant == 0

        mat_data = scipy.io.loadmat(self.input_files[file_idx])
        var_names = [k for k in mat_data.keys() if not k.startswith('__')]
        if not var_names:
            raise ValueError(f"No data variables found in {self.input_files[file_idx]}")
        struct_data = mat_data[var_names[0]]
        var_fields = struct_data.dtype.names
        input_tensors = []
        fvs = struct_data[var_fields[0]][0, 0]

        if self.in_channels == 1:
            amplitude = np.asarray(fvs[-1], dtype=np.float32)
            hidden = np.zeros(amplitude.shape[0], dtype=bool)
            if not is_clean:
                if self.noise_std is not None:
                    amplitude = self._add_gaussian_noise(amplitude)
                if self.mask is not None:
                    amplitude, hidden = self._apply_mask(amplitude, True, idx)
            fvs[-1] = amplitude
            main_input_tensor_fvs = torch.tensor(fvs[-1], dtype=torch.float32).unsqueeze(0)
        else:
            frequency, phase_velocity, amplitude = fvs[0], fvs[1], fvs[-1]
            # Per-frequency normalization: every VISIBLE row has a maximum of exactly 1.0.
            amplitude = amplitude / amplitude.max(axis=1, keepdims=True)

            hidden = np.zeros(amplitude.shape[0], dtype=bool)
            if not is_clean:
                if self.noise_std is not None:
                    amplitude = self._add_gaussian_noise(amplitude)
                if self.mask is not None:
                    # The frequency grid tells us whether row 0 is 5 Hz or 80 Hz.
                    ascending = float(frequency[0, 0]) <= float(frequency[-1, 0])
                    amplitude, hidden = self._apply_mask(amplitude, ascending, idx)

            fvs[-1] = amplitude
            main_input_tensor_fvs = torch.tensor(fvs, dtype=torch.float32)

        if self.return_mask_channel:
            validity = torch.ones(main_input_tensor_fvs.shape[-2:], dtype=torch.float32)
            validity[torch.from_numpy(hidden), :] = 0.0
            main_input_tensor_fvs = torch.cat(
                [main_input_tensor_fvs, validity.unsqueeze(0)], dim=0)

        input_tensors.append(main_input_tensor_fvs)

        if len(self.in_instances) > 1:
            for jj in range(1, len(self.in_instances)):
                add_data = struct_data[var_fields[jj]][0, 0]
                add_tensor = torch.tensor(add_data, dtype=torch.float32).unsqueeze(0)
                if add_tensor.dim() == 1:
                    add_tensor = add_tensor.unsqueeze(0)
                input_tensors.append(add_tensor)

        if self.output_dir is None:  # measured data, no target
            return tuple(input_tensors)

        # Indexed by FILE: all variants of a file share the same Vs profile.
        output_data = pd.read_csv(self.output_files[file_idx], header=None).values
        output_tensor = torch.tensor(output_data, dtype=torch.float32).squeeze()
        return *input_tensors, output_tensor


if __name__ == "__main__":
    in_instances = ['fvs', 'x0', 'dx', 'Ch']
    input_dir = 'training_data/dataset2/input'
    output_dir = 'training_data/dataset2/output100'

    for add_noise in [None,
                      [("gaussian", 0.02)],
                      [("mask", 2, 15.0, 60.0)],
                      [("gaussian", 0.02), ("mask", 2, 15.0, 60.0)]]:
        ds = MyDataset(in_instances, 3, input_dir, output_dir,
                       add_noise=add_noise, return_mask_channel=True)
        print(f"{str(add_noise):60s} -> {ds.desc:55s} len={len(ds)}")
        for k in range(ds.n_variants):
            amp = ds[k][0][2].numpy()
            f_low, f_high = ds.mask_bounds(k)
            n_hidden = int((amp.min(axis=1) < 0).sum())
            band = "full band" if f_low is None else f"visible ({f_low:.1f}, {f_high:.1f}) Hz"
            print(f"     variant {k}: {n_hidden:2d}/76 hidden, {band}")
