"""
model_test_geo.py
=================
Same usage convention as model.py: build the model in main.py, add it to the
`models` list, launch run_train().

    from model_test_geo import (ModelCNN_fvs_v3_geo,
                                ModelDenseNet121_fvs_geo,
                                ModelEfficientNetB0_fvs_geo)

    set_seed(SEED)
    Dense_geo = ModelDenseNet121_fvs_geo(IN_INSTANCES, IN_CHANNELS, 64, 128, fusion_seed=SEED)
    set_seed(SEED)
    Eff_geo = ModelEfficientNetB0_fvs_geo(IN_INSTANCES, IN_CHANNELS, 64, 128, fusion_seed=SEED)

    models = [
        (Dense_geo, "ModelDenseNet121_fvs_geo_90k_RMSELoss"),
        (Eff_geo, "ModelEfficientNetB0_fvs_geo_90k_RMSELoss"),
    ]

Three models are available, all sharing the exact same geometry branch (the G3
configuration: physics features + Fourier encoding + low-rank FiLM). Only the
image backbone changes, and each backbone is copied verbatim from its
counterpart in model.py so the comparison against your existing runs stays
controlled:

    ModelCNN_fvs_v3_geo          <- ModelCNN_fvs            (resnet50,        1000-d)
    ModelDenseNet121_fvs_geo     <- ModelDenseNet121_fvs    (densenet121,     1024-d)
    ModelEfficientNetB0_fvs_geo  <- ModelEfficientNetB0_fvs (efficientnet_b0, 1280-d)

Note that the FiLM generator is sized to each backbone's own output width, and
that DenseNet / EfficientNet use `fusion = Sequential(Linear, LeakyReLU)` while
ModelCNN_fvs uses a bare `Linear` -- both reproduced as in model.py.

-------------------------------------------------------------------------------
SCOPE
-------------------------------------------------------------------------------
The image branch is byte-for-byte the baseline's (ModelCNN_fvs):

    resnet50(pretrained) with conv1 -> Conv2d(in_channels, 64, 7, 2, 3, bias=False)
                              and fc -> Linear(2048, 1000)
    fusion     = Linear(1000 + 128, 1024)
    prediction = the same 4-layer head

Nothing about the image path, the fusion or the head is touched, so any change
in validation loss is attributable to the geometry branch alone.

-------------------------------------------------------------------------------
WHY THE BASELINE'S 30M-PARAMETER GEOMETRY BRANCH IS BADLY CONDITIONED
-------------------------------------------------------------------------------
The baseline does:

    near_offset = x0.repeat(1, 1, 76, 191)      # 14,516 copies of ONE number
    Flatten() -> Linear(14516, 2048) -> ... -> Linear(512, 128)

Write the first layer out. With x_flat = [x0, x0, ..., x0]:

    y_i = sum_j W_ij * x0 + b_i = x0 * (sum_j W_ij) + b_i

The layer is therefore *mathematically identical* to Linear(1, 2048) with the
single weight w_i = sum_j W_ij. The 14,516 copies add exactly zero expressive
power: the whole 30.8M-parameter branch only ever traces a 1-D curve in R^128
parameterized by x0 -- which a 9k-parameter MLP traces just as well.
(`python model_test_geo.py` verifies this numerically.)

But the redundancy is not harmless, it breaks the optimization:

    dL/dW_ij = delta_i * x0     -- the SAME value for all 14,516 columns j

Every copy receives an identical gradient, so the quantity that actually matters,
w_i = sum_j W_ij, moves ~14,516x faster than a single weight would. The effective
learning rate on the geometry pathway is roughly four orders of magnitude above
the one set in main.py (lr=1e-4). That branch is not learning at 1e-4, it is
learning at ~1. Initialization scale and weight decay are distorted the same way.

So the goal is not only "fewer parameters". It is a geometry pathway that
actually trains at the learning rate you chose.

-------------------------------------------------------------------------------
PARAMETER BUDGET (geometry branch only)
-------------------------------------------------------------------------------
    ModelCNN_fvs (baseline)                     30,845,568     53% of the model
    G1  num_fourier_bands=0, use_film=False          9,216     x3347 smaller
    G2  num_fourier_bands=4, use_film=False         13,312     x2317 smaller
    G3  num_fourier_bands=4, use_film=True          49,376      x625 smaller

    G1 vs baseline answers "is 30M parameters buying anything?"
    G2 vs G1 isolates the Fourier encoding.
    G3 vs G2 isolates multiplicative conditioning vs plain concatenation.
"""

import math
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ===========================================================================
# Dataset constants (from the "Training structure example" slide)
# ===========================================================================
DX_VALUES = (1.0, 1.5, 2.0)          # m, receiver spacing
CH_VALUES = (24.0, 48.0)             # number of receivers
L_MIN, L_MAX = 23.0, 94.0            # array length L = (Ch-1)*dx : (24-1)*1 .. (48-1)*2
X0_MIN, X0_MAX = 2.0, 20.0           # m, physical near offset


# ===========================================================================
# 1. Geometry features -- no learnable parameters at all
# ===========================================================================
class GeometryFeatures(nn.Module):
    """(x0', dx, Ch) -> a normalized, physics-aware feature vector.

    In the normalized (f', Vs') plane of the FVS image, every geometry parameter
    draws a straight line through the origin, and it is the SLOPE that carries
    the meaning:

        aperture limit    lambda' = lambda/L = 1     ->  Vs' = f'            slope 1
        spatial aliasing  Vph > 2*dx*f               ->  Vs' = 2 f'/(Ch-1)   slope 2/(Ch-1)
        near-field limit  lambda' < 2 x0'            ->  Vs' = 2 x0' f'      slope 2*x0'

    (the dx of the aliasing slope cancels against L = (Ch-1)*dx, which is why
    that slope depends on Ch alone once the axes are normalized)

    Those slopes are handed to the network directly, together with the physical
    near offset x0 = x0' * L -- a product of two inputs, which a 2-layer MLP
    approximates poorly.

    dx and Ch are also one-hot encoded: they take 3 and 2 distinct values, so
    feeding them as plain floats forces the first Linear to fit a straight line
    through 3 (resp. 2) points. The continuous version is kept alongside so the
    model can still interpolate if the dataset is later extended to new dx / Ch.

    Fourier encoding (num_fourier_bands > 0): an MLP fed a handful of raw scalars
    is spectrally biased -- it fits smooth, low-frequency functions of its input
    and struggles with anything sharper. Sine/cosine octave encoding (Tancik
    et al., 2020) removes that bias without a single parameter, and this is
    exactly the low-input-dimension regime where it helps most.
    Risk: too many octaves invites overfitting on the 114 distinct geometries of
    the dataset (19 x0 values x 3 dx x 2 Ch). Four is deliberately modest.

    All 13 raw columns land in [0, 1] over the full 114-geometry grid.
    """

    CONTINUOUS_NAMES = [
        "x0_norm",         # x0 / L, as stored on disk
        "L_norm",          # aperture -> max resolvable wavelength -> max depth
        "dx_norm",         # receiver spacing
        "Ch_norm",         # receiver count -> spectrum SNR and sidelobe level
        "x0_phys_norm",    # x0 in metres = x0_norm * L (the product an MLP cannot form)
        "x_far_norm",      # x0 + L, offset of the last receiver
        "alias_slope",     # 2 / (Ch - 1), slope of the aliasing line
        "x0_over_dx",      # x0 / dx = x0' * (Ch - 1): ratio of the near-field slope
                           # to the aliasing slope, i.e. how many receiver spacings
                           # fit inside the near offset. Genuinely non-linear in the
                           # inputs, unlike 2*x0' which is a rescaling of x0_norm.
    ]
    CATEGORICAL_NAMES = ["dx=1.0", "dx=1.5", "dx=2.0", "Ch=24", "Ch=48"]

    def __init__(self, num_fourier_bands=4):
        super().__init__()
        self.num_fourier_bands = num_fourier_bands
        self.register_buffer("dx_values", torch.tensor(DX_VALUES))
        self.register_buffer("ch_values", torch.tensor(CH_VALUES))
        if num_fourier_bands > 0:
            # Octave frequencies 1, 2, 4, 8 (times pi) over inputs living in [0, 1]
            self.register_buffer("bands", 2.0 ** torch.arange(num_fourier_bands) * math.pi)

    @property
    def out_dim(self):
        return (len(self.CONTINUOUS_NAMES) * (1 + 2 * self.num_fourier_bands)
                + len(self.CATEGORICAL_NAMES))

    @property
    def feature_names(self):
        """Column names of the raw (pre-Fourier) vector."""
        return self.CONTINUOUS_NAMES + self.CATEGORICAL_NAMES

    @staticmethod
    def _one_hot(value, table):
        """Nearest-value one-hot, robust to the float rounding of the .mat files."""
        idx = (value - table.view(1, -1)).abs().argmin(dim=1)
        return torch.nn.functional.one_hot(idx, num_classes=table.numel()).to(value.dtype)

    def raw(self, x0_n, dx, ch):
        """(B, 13): 8 continuous features in [0, 1] followed by 5 one-hot bits."""
        L = (ch - 1.0) * dx
        x0_phys = x0_n * L
        x_far = x0_phys + L

        cont = torch.cat([
            x0_n,
            (L - L_MIN) / (L_MAX - L_MIN),
            (dx - DX_VALUES[0]) / (DX_VALUES[-1] - DX_VALUES[0]),
            (ch - CH_VALUES[0]) / (CH_VALUES[-1] - CH_VALUES[0]),
            (x0_phys - X0_MIN) / (X0_MAX - X0_MIN),
            (x_far - (X0_MIN + L_MIN)) / ((X0_MAX + L_MAX) - (X0_MIN + L_MIN)),
            (2.0 / (ch - 1.0).clamp_min(1.0)) / (2.0 / (CH_VALUES[0] - 1.0)),
            (x0_phys / dx.clamp_min(1e-6)) / (X0_MAX / DX_VALUES[0]),
        ], dim=1)

        return torch.cat([cont,
                          self._one_hot(dx, self.dx_values),
                          self._one_hot(ch, self.ch_values)], dim=1)

    def forward(self, x0_n, dx, ch):
        raw = self.raw(x0_n, dx, ch)
        if self.num_fourier_bands == 0:
            return raw
        n_cont = len(self.CONTINUOUS_NAMES)
        cont, cat = raw[:, :n_cont], raw[:, n_cont:]
        proj = cont.unsqueeze(-1) * self.bands              # (B, n_cont, J)
        return torch.cat([cont, proj.sin().flatten(1), proj.cos().flatten(1), cat], dim=1)


# ===========================================================================
# 2. Low-rank FiLM on the 1000-d image vector
# ===========================================================================
class LowRankFiLM(nn.Module):
    """f' = (1 + gamma) * f + beta, with (gamma, beta) built through a rank-r bottleneck.

    Plain concatenation lets the fusion layer compute W_img*f_img + W_geo*f_geo:
    the geometry can only ADD a bias to the image features, never modulate them.
    But "this region of the spectrum is aliased" is multiplicative by nature.
    FiLM provides that interaction.

    A full Linear(128, 2000) would cost 258k parameters to condition on 114
    discrete geometries -- easy to overfit -- so it is factorized through a
    rank-r bottleneck (r=16 -> 34k).

    Zero-init on the output layer => gamma = beta = 0 at step 0, so training
    starts strictly identical to the concat-only variant and the modulation is
    learned progressively. No risk of destabilizing the pretrained backbone.
    """

    def __init__(self, cond_dim, num_features, rank=16):
        super().__init__()
        self.down = nn.Linear(cond_dim, rank)
        self.up = nn.Linear(rank, 2 * num_features)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feat, cond):
        gamma, beta = self.up(self.down(cond)).chunk(2, dim=1)
        return (1.0 + gamma) * feat + beta


# ===========================================================================
# 3. The model
# ===========================================================================
class ModelCNN_fvs_v3_geo(nn.Module):
    """Baseline image branch + a compact, physics-aware geometry branch.

    Signature matches the rest of the family:
        ModelCNN_fvs_v3_geo(IN_INSTANCES, IN_CHANNELS, geo_hidden, geo_feat_dim,
                            fusion_seed=SEED)

    Variants to compare (everything else identical):
        G1  num_fourier_bands=0, use_film=False        9,216 params
        G2  num_fourier_bands=4, use_film=False       13,312 params
        G3  num_fourier_bands=4, use_film=True        49,376 params

    geo_feat_dim MUST stay at 128, otherwise `fusion` no longer has the baseline
    shape and this stops being a controlled experiment.

    fusion_seed works as in ModelCNN_fvs_v2_film / _all_geo / _all_geo_film: the
    geometry branch consumes a different number of RNG draws than the baseline's,
    so without reseeding, fusion and prediction would start from different
    weights and part of the measured difference would just be initialization.
    """

    def __init__(self, in_instances, in_channels=3, geo_hidden=64, geo_feat_dim=128,
                 num_fourier_bands=4, use_film=True, film_rank=16,
                 out_dim=101, fusion_seed=None):
        super().__init__()
        self.in_instances = in_instances

        # ---------- Image branch: copied verbatim from ModelCNN_fvs ----------
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                          padding=3, bias=False)
        self.base_model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        # Note: reusing the pretrained conv1 when in_channels == 3 would be a free
        # win (the shape is identical, yet the layer is re-initialized from scratch
        # here, as in the baseline). Deliberately NOT done: it would change the
        # backbone and contaminate this experiment. Test it separately, on top of
        # whichever geometry branch wins.

        # ---------- Geometry branch: the only thing that changes ----------
        self.geo_features = GeometryFeatures(num_fourier_bands=num_fourier_bands)
        self.geo_encoder = nn.Sequential(
            nn.Linear(self.geo_features.out_dim, geo_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(geo_hidden, geo_feat_dim),
            nn.LeakyReLU(0.1),
        )
        # No BatchNorm1d here, unlike ModelCNN_fvs_v2: with batch_size=8 it would
        # estimate mean/var from 8 samples of variables taking 2 or 3 distinct
        # values, injecting batch-dependent noise into a branch that carries an
        # exact, noise-free measurement. Every feature above is already
        # deterministically normalized using known dataset ranges.

        self.film = LowRankFiLM(geo_feat_dim, 1000, rank=film_rank) if use_film else None

        # ---------- Fusion + head: copied verbatim from ModelCNN_fvs ----------
        if fusion_seed is not None:
            torch.manual_seed(fusion_seed)
        self.fusion = nn.Linear(1000 + geo_feat_dim, 1024)
        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_dim),
        )

    @staticmethod
    def _as_column(t):
        """Whatever shape the dataset produced -> (B, 1) float."""
        return t.reshape(t.size(0), -1)[:, :1].float()

    def geometry_branch_parameters(self):
        """Parameter count of the geometry pathway only, for the comparison table."""
        mods = [self.geo_encoder] + ([self.film] if self.film is not None else [])
        return sum(p.numel() for m in mods for p in m.parameters())

    def forward(self, *inputs):
        fvs = inputs[0]
        x0_n = self._as_column(inputs[1])   # index 1 = 'x0', same convention as the baseline
        dx = self._as_column(inputs[2])
        ch = self._as_column(inputs[3])

        fvs_features = self.base_model(fvs)                      # (B, 1000), unchanged
        geo_features = self.geo_encoder(self.geo_features(x0_n, dx, ch))   # (B, 128)

        if self.film is not None:
            fvs_features = self.film(fvs_features, geo_features)

        x = self.fusion(torch.cat((fvs_features, geo_features), dim=1))
        return self.prediction(x)


# ===========================================================================
# 4. Same geometry branch, other backbones
# ===========================================================================
# ModelCNN_fvs_v3_geo above is deliberately left untouched: its state_dict keys
# must keep matching the G3 checkpoint already trained. The two classes below
# therefore repeat the geometry/fusion/head code rather than factoring it into a
# shared base class. A little duplication is a cheap price for not breaking a
# checkpoint that cost 10 hours of GPU time.


def _as_column(t):
    """Whatever shape the dataset produced -> (B, 1) float."""
    return t.reshape(t.size(0), -1)[:, :1].float()


class ModelDenseNet121_fvs_geo(nn.Module):
    """DenseNet-121 image branch (verbatim from ModelDenseNet121_fvs) + the G3 geometry branch.

    DenseNet concatenates every previous feature map inside a block, so channels
    are reused rather than recomputed. That gives strong feature propagation at
    a modest parameter count (~7M for the backbone), which is why it holds up
    well on the FVS images despite being much smaller than resnet50.

    Backbone output is 1024-d (not 1000 as in ModelCNN_fvs), so the FiLM
    generator is sized accordingly. `fusion` is Sequential(Linear, LeakyReLU)
    here, exactly as in model.py -- ModelCNN_fvs uses a bare Linear instead.
    """

    def __init__(self, in_instances, in_channels=3, geo_hidden=64, geo_feat_dim=128,
                 num_fourier_bands=4, use_film=True, film_rank=16,
                 out_dim=101, fusion_seed=None):
        super().__init__()
        self.in_instances = in_instances

        # ---------- Image branch: copied verbatim from ModelDenseNet121_fvs ----------
        self.base_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.base_model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                                   stride=2, padding=3, bias=False)
        self.base_model.classifier = nn.Identity()
        backbone_out_features = 1024

        # ---------- Geometry branch: identical to the G3 configuration ----------
        self.geo_features = GeometryFeatures(num_fourier_bands=num_fourier_bands)
        self.geo_encoder = nn.Sequential(
            nn.Linear(self.geo_features.out_dim, geo_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(geo_hidden, geo_feat_dim),
            nn.LeakyReLU(0.1),
        )
        self.film = (LowRankFiLM(geo_feat_dim, backbone_out_features, rank=film_rank)
                     if use_film else None)

        # ---------- Fusion + head: copied verbatim from ModelDenseNet121_fvs ----------
        if fusion_seed is not None:
            torch.manual_seed(fusion_seed)
        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_features + geo_feat_dim, 1024),
            nn.LeakyReLU(0.1),
        )
        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_dim),
        )

    def geometry_branch_parameters(self):
        mods = [self.geo_encoder] + ([self.film] if self.film is not None else [])
        return sum(p.numel() for m in mods for p in m.parameters())

    def forward(self, *inputs):
        fvs = inputs[0]
        x0_n = _as_column(inputs[1])   # index 1 = 'x0', same convention as the baseline
        dx = _as_column(inputs[2])
        ch = _as_column(inputs[3])

        fvs_features = self.base_model(fvs)                              # (B, 1024)
        geo_features = self.geo_encoder(self.geo_features(x0_n, dx, ch))  # (B, 128)

        if self.film is not None:
            fvs_features = self.film(fvs_features, geo_features)

        x = self.fusion(torch.cat((fvs_features, geo_features), dim=1))
        return self.prediction(x)


class ModelEfficientNetB0_fvs_geo(nn.Module):
    """EfficientNet-B0 image branch (verbatim from ModelEfficientNetB0_fvs) + the G3 geometry branch.

    Depthwise-separable MBConv blocks with squeeze-and-excitation gating, ending
    in a 1x1 conv to 1280 channels before pooling. By far the smallest backbone
    of the family (~5.3M parameters), which makes this the cheapest run of the
    three -- useful when GPU time is the binding constraint.

    Note: setting `classifier = nn.Identity()` also removes EfficientNet's own
    dropout layer. That is what ModelEfficientNetB0_fvs does, and it is
    reproduced here so the two runs stay comparable -- but it does mean this
    model has no regularization at all, which matters given the ~10x train/val
    gap already visible in the logs.
    """

    def __init__(self, in_instances, in_channels=3, geo_hidden=64, geo_feat_dim=128,
                 num_fourier_bands=4, use_film=True, film_rank=16,
                 out_dim=101, fusion_seed=None):
        super().__init__()
        self.in_instances = in_instances

        # ---------- Image branch: copied verbatim from ModelEfficientNetB0_fvs ----------
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.base_model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3,
                                                   stride=2, padding=1, bias=False)
        self.base_model.classifier = nn.Identity()
        backbone_out_features = 1280

        # ---------- Geometry branch: identical to the G3 configuration ----------
        self.geo_features = GeometryFeatures(num_fourier_bands=num_fourier_bands)
        self.geo_encoder = nn.Sequential(
            nn.Linear(self.geo_features.out_dim, geo_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(geo_hidden, geo_feat_dim),
            nn.LeakyReLU(0.1),
        )
        self.film = (LowRankFiLM(geo_feat_dim, backbone_out_features, rank=film_rank)
                     if use_film else None)

        # ---------- Fusion + head: copied verbatim from ModelEfficientNetB0_fvs ----------
        if fusion_seed is not None:
            torch.manual_seed(fusion_seed)
        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_features + geo_feat_dim, 1024),
            nn.LeakyReLU(0.1),
        )
        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, out_dim),
        )

    def geometry_branch_parameters(self):
        mods = [self.geo_encoder] + ([self.film] if self.film is not None else [])
        return sum(p.numel() for m in mods for p in m.parameters())

    def forward(self, *inputs):
        fvs = inputs[0]
        x0_n = _as_column(inputs[1])   # index 1 = 'x0', same convention as the baseline
        dx = _as_column(inputs[2])
        ch = _as_column(inputs[3])

        fvs_features = self.base_model(fvs)                              # (B, 1280)
        geo_features = self.geo_encoder(self.geo_features(x0_n, dx, ch))  # (B, 128)

        if self.film is not None:
            fvs_features = self.film(fvs_features, geo_features)

        x = self.fusion(torch.cat((fvs_features, geo_features), dim=1))
        return self.prediction(x)


# ===========================================================================
# 5. Self-test: rank-1 proof, parameter budget, shape check
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 4, 76, 191
    in_instances = ['fvs', 'x0', 'dx', 'Ch']

    dx = torch.tensor([[1.0], [1.5], [2.0], [2.0]])
    ch = torch.tensor([[24.0], [48.0], [24.0], [48.0]])
    L = (ch - 1) * dx
    x0_n = torch.tensor([[2.0], [10.0], [20.0], [5.0]]) / L
    fvs = torch.rand(B, 3, H, W)

    # (a) the baseline geometry branch is mathematically Linear(1, 2048)
    baseline_branch = nn.Sequential(
        nn.Flatten(), nn.Linear(14516, 2048), nn.LeakyReLU(),
        nn.Linear(2048, 512), nn.LeakyReLU(), nn.Linear(512, 128))
    n_baseline = sum(p.numel() for p in baseline_branch.parameters())
    lin = baseline_branch[1]
    x0_repeated = x0_n.view(B, 1, 1, 1).repeat(1, 1, H, W)       # what ModelCNN_fvs does
    with torch.no_grad():
        actual = lin(x0_repeated.flatten(1))
        equivalent = x0_n * lin.weight.sum(dim=1).view(1, -1) + lin.bias
    print("Linear(14516, 2048) on 14,516 copies of x0   vs   x0 * sum_j(W_ij) + b")
    print(f"   max |difference| = {(actual - equivalent).abs().max():.3e}"
          "  -> no expressive power beyond Linear(1, 2048)\n")

    # (b) geometry-branch ablation on the resnet50 backbone
    variants = {
        "G1 physics features only":   dict(num_fourier_bands=0, use_film=False),
        "G2 + Fourier encoding":      dict(num_fourier_bands=4, use_film=False),
        "G3 + low-rank FiLM (r=16)":  dict(num_fourier_bands=4, use_film=True),
    }
    print(f"{'geometry branch (resnet50)':<38}{'geo branch':>14}{'total':>12}{'vs baseline':>13}")
    print(f"   {'ModelCNN_fvs (baseline)':<35}{n_baseline:>14,}{58.26:>11.2f}M{'1x':>13}")
    for name, cfg in variants.items():
        m = ModelCNN_fvs_v3_geo(in_instances, 3, **cfg).eval()
        with torch.no_grad():
            out = m(fvs, x0_n, dx, ch)
        assert out.shape == (B, 101), out.shape
        g = m.geometry_branch_parameters()
        tot = sum(p.numel() for p in m.parameters())
        print(f"   {name:<35}{g:>14,}{tot/1e6:>11.2f}M{n_baseline/g:>12.0f}x")

    # (c) the three backbones, all with the same G3 geometry branch
    print(f"\n{'G3 geometry branch, other backbones':<38}{'geo branch':>14}{'total':>12}"
          f"{'backbone out':>14}")
    backbones = [
        ("resnet50", ModelCNN_fvs_v3_geo),
        ("densenet121", ModelDenseNet121_fvs_geo),
        ("efficientnet_b0", ModelEfficientNetB0_fvs_geo),
    ]
    for name, cls in backbones:
        m = cls(in_instances, 3).eval()
        with torch.no_grad():
            out = m(fvs, x0_n, dx, ch)
        assert out.shape == (B, 101), (name, out.shape)
        g = m.geometry_branch_parameters()
        tot = sum(p.numel() for p in m.parameters())
        width = m.film.up.out_features // 2 if m.film is not None else None
        print(f"   {name:<35}{g:>14,}{tot/1e6:>11.2f}M{width:>14}")

    # (d) fusion / head shapes must match their model.py counterparts
    m = ModelCNN_fvs_v3_geo(in_instances, 3)
    print(f"\nresnet50        fusion {tuple(m.fusion.weight.shape)}"
          f"   (ModelCNN_fvs: (1024, 1128), bare Linear)")
    m = ModelDenseNet121_fvs_geo(in_instances, 3)
    print(f"densenet121     fusion {tuple(m.fusion[0].weight.shape)}"
          f"   (ModelDenseNet121_fvs: (1024, 1152))")
    m = ModelEfficientNetB0_fvs_geo(in_instances, 3)
    print(f"efficientnet_b0 fusion {tuple(m.fusion[0].weight.shape)}"
          f"   (ModelEfficientNetB0_fvs: (1024, 1408))")
    print(f"head output     {tuple(m.prediction[-1].weight.shape)}   (all: (101, 128))")

    # (d) every raw feature stays in [0, 1] over the whole 114-geometry grid
    gf = GeometryFeatures(num_fourier_bands=0)
    rows = [gf.raw(torch.tensor([[x0 / ((c - 1) * d)]]),
                   torch.tensor([[d]]), torch.tensor([[c]]))
            for x0 in range(2, 21) for d in DX_VALUES for c in CH_VALUES]
    R = torch.cat(rows)
    print(f"\n{len(R)} geometries -- raw feature ranges:")
    for j, name in enumerate(gf.feature_names):
        print(f"   {name:<14} [{R[:, j].min():.3f}, {R[:, j].max():.3f}]")
