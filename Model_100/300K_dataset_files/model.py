import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kinh DL model Architecture  (Baseline)
class ModelCNN_fvs(nn.Module):

    def __init__(self, in_instances, in_channels=1):
        super(ModelCNN_fvs, self).__init__()
        self.in_instances = in_instances

        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)

        self.layout_feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear( 14516, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
        )

        self.fusion = nn.Linear(1000 + 128, 1024)

        # Prediction layers
        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 101),
        )

    def forward(self, *inputs):
        # Call input feature
        fvs = inputs[0]
        near_offset = inputs[1]  # index 1 = 'x0' cuz in_instances = ['fvs', 'x0', 'dx', 'Ch'] now
        near_offset = near_offset.repeat(1, 1, fvs.size(2), fvs.size(3))

        # Feature extraction
        fvs_features = self.base_model(fvs)
        layout_features = self.layout_feature_extractor(near_offset)
        combined_features = torch.cat((fvs_features, layout_features), dim=1)
        x = self.fusion(combined_features)

        # Prediction
        out = self.prediction(x)

        return out

# Same model as Dr. Kinh's one, BUT without 30M parameters for representing x0
# MLP for x0
# 2 tests needed :
#   - x0_hidden = 32 and x0_feat_dim = 128 (default)
#   - x0_hidden = 16 and x0_feat_dim = 16 (fewer parameters)
#   - OPTIONAL : x0_hidden = 32 and x0_feat_dim = 16 (fewer parameters)
class ModelCNN_fvs_v2(nn.Module):

    def __init__(self, in_instances, in_channels=1, x0_hidden=32, x0_feat_dim=128):
        super().__init__()
        self.in_instances = in_instances

        # --- Image branch: identical to the original baseline ---
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)

        # x0 branch: a real scalar encoder, not a spatial broadcast
        self.x0_encoder = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(1, x0_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(x0_hidden, x0_feat_dim),
            nn.LeakyReLU(0.1),
        )

        # --- Fusion + prediction: byte-for-byte identical to the original ---
        self.fusion = nn.Linear(1000 + x0_feat_dim, 1024)
        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 101),
        )

    def forward(self, *inputs):
        fvs = inputs[0]
        x0 = inputs[1]  # index 1 = 'x0', same convention as the original baseline

        # Flatten x0 to (B, 1). No repeat over H*W, no Flatten() of a fake grid.
        x0 = x0.reshape(x0.size(0), -1)

        fvs_features = self.base_model(fvs)  # (B, 1000)
        x0_features = self.x0_encoder(x0)  # (B, 128)
        combined_features = torch.cat((fvs_features, x0_features), dim=1)
        x = self.fusion(combined_features)

        out = self.prediction(x)
        return out

# Old name : ModelCNN_fvs
# SCL3 Version
class ModelResNet50_fvs(nn.Module):
    """ResNet-50 backbone (native 2048-dim output) fused with MASW layout features."""

    def __init__(self, in_instances, in_channels=1):
        super().__init__()
        self.in_instances = in_instances

        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Identity()
        backbone_out_features = 2048

        self.layout_feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14516, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_features + 128, 1024),
            nn.LeakyReLU(0.1),
        )

        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 101),
        )

    def forward(self, *inputs):
        fvs = inputs[0]
        near_offset = inputs[1]  # index 1 = 'x0' cuz in_instances = ['fvs', 'x0', 'dx', 'Ch'] now
        near_offset = near_offset.repeat(1, 1, fvs.size(2), fvs.size(3))

        fvs_features = self.base_model(fvs)
        layout_features = self.layout_feature_extractor(near_offset)
        combined_features = torch.cat((fvs_features, layout_features), dim=1)
        x = self.fusion(combined_features)

        out = self.prediction(x)
        return out

# Second test, x0 fewer parameters (not 30M), and default backbone out features for resnet-50
class ModelResNet50_fvs_v2(nn.Module):
    """ResNet-50 backbone (native 2048-dim output) fused with MASW layout features."""

    def __init__(self, in_instances, in_channels=1, x0_hidden=32, x0_feat_dim=128):
        super().__init__()
        self.in_instances = in_instances

        # --- Image branch: identical to the original baseline ---
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Identity()
        backbone_out_features = 2048

        # x0 branch: a real scalar encoder, not a spatial broadcast
        self.x0_encoder = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(1, x0_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(x0_hidden, x0_feat_dim),
            nn.LeakyReLU(0.1),
        )

        # --- Fusion + prediction: byte-for-byte identical to the original ---
        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_features + x0_feat_dim, 1024),
            nn.LeakyReLU(0.1),
        )

        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 101),
        )

    def forward(self, *inputs):
        fvs = inputs[0]
        x0 = inputs[1]  # index 1 = 'x0' cuz in_instances = ['fvs', 'x0', 'dx', 'Ch'] now

        # Flatten x0 to (B, 1). No repeat over H*W, no Flatten() of a fake grid.
        x0 = x0.reshape(x0.size(0), -1)

        fvs_features = self.base_model(fvs)  # (B, 2048)  Identity
        x0_features = self.x0_encoder(x0)  # (B, 128)
        combined_features = torch.cat((fvs_features, x0_features), dim=1)
        x = self.fusion(combined_features)

        out = self.prediction(x)
        return out

# !!! DROPPED !!!
# This model does not represent well the FVS image
class ModelResNet34_fvs(nn.Module):
    """ResNet-34 backbone (native 512-dim output) fused with MASW layout features."""

    def __init__(self, in_instances, in_channels=1):
        super().__init__()
        self.in_instances = in_instances

        self.base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Identity()
        backbone_out_features = 512

        self.layout_feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14516, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_features + 128, 1024),
            nn.LeakyReLU(0.1),
        )

        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 101),
        )

    def forward(self, *inputs):
        fvs = inputs[0]
        near_offset = inputs[1]  # index 1 = 'x0' cuz in_instances = ['fvs', 'x0', 'dx', 'Ch'] now
        near_offset = near_offset.repeat(1, 1, fvs.size(2), fvs.size(3))

        fvs_features = self.base_model(fvs)
        layout_features = self.layout_feature_extractor(near_offset)
        combined_features = torch.cat((fvs_features, layout_features), dim=1)
        x = self.fusion(combined_features)

        out = self.prediction(x)
        return out


class ModelDenseNet121_fvs(nn.Module):
    """DenseNet-121 backbone (native 1024-dim output) fused with MASW layout features."""

    def __init__(self, in_instances, in_channels=1):
        super().__init__()
        self.in_instances = in_instances

        self.base_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.base_model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.classifier = nn.Identity()
        backbone_out_features = 1024

        self.layout_feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14516, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_features + 128, 1024),
            nn.LeakyReLU(0.1),
        )

        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 101),
        )

    def forward(self, *inputs):
        fvs = inputs[0]
        near_offset = inputs[1]  # index 1 = 'x0' cuz in_instances = ['fvs', 'x0', 'dx', 'Ch'] now
        near_offset = near_offset.repeat(1, 1, fvs.size(2), fvs.size(3))

        fvs_features = self.base_model(fvs)
        layout_features = self.layout_feature_extractor(near_offset)
        combined_features = torch.cat((fvs_features, layout_features), dim=1)
        x = self.fusion(combined_features)

        out = self.prediction(x)
        return out


class ModelSwinT_fvs(nn.Module):
    """Swin-T backbone (native 768 output) fused with MASW layout features.

    swin_t: patch_size=4, embed_dim=96, depths=[2,2,6,2] -> native pooled
    features = 96 * 2**3 = 768. The backbone does not need a fixed input
    resolution: shifted_window_attention PADS the feature map to a multiple
    of the window size (7) internally at every stage, so the raw FVS spatial
    size can be fed directly without resizing.
    """

    def __init__(self, in_instances, in_channels=1):
        super().__init__()
        self.in_instances = in_instances

        self.base_model = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.base_model.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=(4, 4), stride=(4, 4))
        self.base_model.head = nn.Identity()
        backbone_out_features = 768

        self.layout_feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14516, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_features + 128, 1024),
            nn.LeakyReLU(0.1),
        )

        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 101),
        )

    def forward(self, *inputs):
        fvs = inputs[0]
        near_offset = inputs[1]  # index 1 = 'x0' cuz in_instances = ['fvs', 'x0', 'dx', 'Ch'] now
        near_offset = near_offset.repeat(1, 1, fvs.size(2), fvs.size(3))

        fvs_features = self.base_model(fvs)
        layout_features = self.layout_feature_extractor(near_offset)
        combined_features = torch.cat((fvs_features, layout_features), dim=1)
        x = self.fusion(combined_features)

        out = self.prediction(x)
        return out


class ModelEfficientNetB0_fvs(nn.Module):
    """EfficientNet-B0 backbone (native 1280 output) fused with MASW layout features.

    efficientnet_b0: MBConv blocks with squeeze-and-excitation, ending in a
    1x1 conv to 1280 channels (4 * 320, the last stage's output channels)
    before pooling. Depthwise-separable convolutions with
    squeeze-and-excitation gating, rather than plain/dense convs or window
    attention. The smallest backbone here by a wide margin (~5.3M
    params vs 8M+ for the others), useful as a lightweight reference point.
    """

    def __init__(self, in_instances, in_channels=1):
        super().__init__()
        self.in_instances = in_instances

        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.base_model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.base_model.classifier = nn.Identity()
        backbone_out_features = 1280

        self.layout_feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14516, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_features + 128, 1024),
            nn.LeakyReLU(0.1),
        )

        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 101),
        )

    def forward(self, *inputs):
        fvs = inputs[0]
        near_offset = inputs[1]  # index 1 = 'x0' cuz in_instances = ['fvs', 'x0', 'dx', 'Ch'] now
        near_offset = near_offset.repeat(1, 1, fvs.size(2), fvs.size(3))

        fvs_features = self.base_model(fvs)
        layout_features = self.layout_feature_extractor(near_offset)
        combined_features = torch.cat((fvs_features, layout_features), dim=1)
        x = self.fusion(combined_features)

        out = self.prediction(x)
        return out

class ResBlock2d(nn.Module):
    """Residual conv block (2x Conv3x3+BN+LeakyReLU + skip connection), then downsampling.

    Used by ModelCustomCNN_fvs below. Unlike the pretrained ImageNet backbones used
    by the classes above, this block trains from scratch. An fvs image (frequency,
    phase velocity, amplitude) has none of the statistics of a natural photo.
    ImageNet transfer adds little here. Training from scratch gives a smaller model,
    tailored to the task.
    """

    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        # 1x1 projection if channel count changes, for the skip connection
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return self.pool(out)


class ModelCustomCNN_fvs(nn.Module):
    """
    "From scratch" alternative to the models above (ModelCNN_fvs / ModelResNetXX_fvs /
    ModelDenseNet121_fvs / ModelSwinT_fvs / ModelEfficientNetB0_fvs). No pretrained
    ImageNet backbone.

    Deliberate differences from the models above:
      1) No ImageNet pretraining. Not very useful for an fvs image (channels =
         frequency / phase velocity / amplitude), which looks nothing like a photo.
         The CNN encoder is a small "home-made" ResNet (4 ResBlock2d), trained from scratch.
      2) x0, dx AND Ch are used. The forward() above only reads inputs[0] (fvs)
         and inputs[1] ('x0'). dx and Ch (inputs[2], inputs[3]) are loaded by the
         dataset but never passed to the model. Here, ALL scalars after 'fvs' are
         concatenated and used (the count is inferred from len(in_instances) - 1,
         so it works with ['fvs','x0'] as well as ['fvs','x0','dx','Ch'] or more).
      3) Scalars go through a small MLP (BatchNorm1d + 2x Linear) instead of being
         repeated over the whole spatial grid (H*W = 14,516 identical values) before
         a Linear(14516, 2048). Repeating a value then flattening adds no information.
         It just inflates dimensionality. But it costs ~29.7M params (14,516 x 2048)
         per scalar encoded that way.
      4) Per-channel input normalization (BatchNorm2d) on fvs. Checked on 3 real
         samples: channel 0 (frequency) has a scale that depends on dx/Ch (max ~7.52
         for dx=2/Ch=48, ~1.84 for dx=1/Ch=24), while channel 2 (amplitude) is always
         in [0,1]. Without normalization, the first Conv2d would see channels on very
         different scales. BatchNorm2d(in_channels) at the start fixes this automatically.
      5) L = (Ch-1)*dx (sensor array length, per your formula) is computed and added
         as an extra scalar when 'dx' and 'Ch' are in in_instances. Stored x0 is
         already x0_real/L. So the network gets both the normalized ratio AND enough
         to reconstruct x0_real, with no information loss. L has a direct physical
         meaning in MASW (correlated with achievable investigation depth), so exposing
         it explicitly makes learning easier.

    Stays compatible with the existing pipeline (train.py / main.py / Call_dataset.py):
    __init__(in_instances, in_channels) and forward(*inputs).
    """

    def __init__(self, in_instances, in_channels=3, base_ch=32, img_feat_dim=256,
                 scalar_feat_dim=64, dropout=0.15):
        super().__init__()
        self.in_instances = in_instances
        n_scalars = len(in_instances) - 1  # everything but 'fvs' (x0, dx, Ch, ...)
        assert n_scalars >= 1, "in_instances must contain 'fvs' + at least one scalar"

        # Locate dx/Ch in in_instances to compute L=(Ch-1)*dx too (if available)
        self._dx_idx = in_instances.index('dx') if 'dx' in in_instances else None
        self._ch_idx = in_instances.index('Ch') if 'Ch' in in_instances else None
        self.use_L_feature = self._dx_idx is not None and self._ch_idx is not None
        n_scalar_features = n_scalars + (1 if self.use_L_feature else 0)

        # --- Image branch: residual CNN from scratch, (B, in_channels, H, W) -> (B, img_feat_dim) ---
        self.input_norm = nn.BatchNorm2d(in_channels)  # normalizes frequency/velocity/amplitude, very different scales
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.layer1 = ResBlock2d(base_ch, base_ch * 2)        # 76x191 -> 38x95
        self.layer2 = ResBlock2d(base_ch * 2, base_ch * 4)    # 38x95  -> 19x47
        self.layer3 = ResBlock2d(base_ch * 4, base_ch * 8)    # 19x47  -> 9x23
        self.layer4 = ResBlock2d(base_ch * 8, img_feat_dim)   # 9x23   -> 4x11
        # AdaptiveAvgPool -> robust if H,W change slightly (no upstream resize needed)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- Scalar branch: x0, dx, Ch (+ derived L, + any other scalar added to in_instances) ---
        self.scalar_encoder = nn.Sequential(
            nn.BatchNorm1d(n_scalar_features),  # normalizes x0/dx/Ch/L despite their very different scales
            nn.Linear(n_scalar_features, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, scalar_feat_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # --- Fusion + regression head (101 outputs: nL thicknesses + (nL+1) Vs) ---
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + scalar_feat_dim, 1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.prediction = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 101),
        )

    def forward(self, *inputs):
        fvs = inputs[0]              # (B, in_channels, H, W)
        scalar_inputs = inputs[1:]   # (x0, dx, Ch, ...), each with 1 value/sample

        # --- Image ---
        x = self.input_norm(fvs)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        img_features = self.gap(x).flatten(1)  # (B, img_feat_dim)

        # --- Scalars: each flattened to (B,1), + L=(Ch-1)*dx if available, then concatenated ---
        scalars_list = [s.reshape(s.size(0), -1) for s in scalar_inputs]
        if self.use_L_feature:
            dx_t = inputs[self._dx_idx].reshape(inputs[self._dx_idx].size(0), -1)
            ch_t = inputs[self._ch_idx].reshape(inputs[self._ch_idx].size(0), -1)
            L = (ch_t - 1) * dx_t
            scalars_list.append(L)
        scalars = torch.cat(scalars_list, dim=1)
        scalar_features = self.scalar_encoder(scalars)

        # --- Fusion + prediction ---
        combined_features = torch.cat((img_features, scalar_features), dim=1)
        x = self.fusion(combined_features)
        out = self.prediction(x)

        return out

if __name__ == '__main__':
    print("non")