import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kinh model (Baseline)
class ModelCNN_fvs(nn.Module):

    def __init__(self, in_instances, in_channels=1):
        super(ModelCNN_fvs, self).__init__()
        self.in_instances = in_instances

        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)

        self.layout_feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(36176, 2048),
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
        near_offset = inputs[2]
        near_offset = near_offset.repeat(1, 1, fvs.size(2), fvs.size(3))

        # Feature extraction
        fvs_features = self.base_model(fvs)
        layout_features = self.layout_feature_extractor(near_offset)
        combined_features = torch.cat((fvs_features, layout_features), dim=1)
        x = self.fusion(combined_features)

        # Prediction
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
            nn.Linear(36176, 2048),
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
        near_offset = inputs[2]
        near_offset = near_offset.repeat(1, 1, fvs.size(2), fvs.size(3))

        fvs_features = self.base_model(fvs)
        layout_features = self.layout_feature_extractor(near_offset)
        combined_features = torch.cat((fvs_features, layout_features), dim=1)
        x = self.fusion(combined_features)

        out = self.prediction(x)
        return out


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
            nn.Linear(36176, 2048),
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
        near_offset = inputs[2]
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
            nn.Linear(36176, 2048),
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
        near_offset = inputs[2]
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
            nn.Linear(36176, 2048),
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
        near_offset = inputs[2]
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
            nn.Linear(36176, 2048),
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
        near_offset = inputs[2]
        near_offset = near_offset.repeat(1, 1, fvs.size(2), fvs.size(3))

        fvs_features = self.base_model(fvs)
        layout_features = self.layout_feature_extractor(near_offset)
        combined_features = torch.cat((fvs_features, layout_features), dim=1)
        x = self.fusion(combined_features)

        out = self.prediction(x)
        return out

if __name__ == '__main__':
    print("non")