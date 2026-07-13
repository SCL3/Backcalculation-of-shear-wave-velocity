import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.models import densenet121, DenseNet121_Weights

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DL Architecture
class _BaseModel_fvs(nn.Module):
    """Shared fusion/prediction head for fvs backbones (base_model set by subclasses)."""

    def __init__(self, in_instances, backbone_out_features=1000):
        super().__init__()
        self.in_instances = in_instances
        self.base_model: nn.Module  # assigned by subclass __init__

        self.layout_feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(36176, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
        )

        self.fusion = nn.Linear(backbone_out_features + 128, 1024)

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


class ModelResNet50_fvs(_BaseModel_fvs):

    def __init__(self, in_instances, in_channels=1):
        super().__init__(in_instances, backbone_out_features=1000)

        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)


class ModelResNet34_fvs(_BaseModel_fvs):

    def __init__(self, in_instances, in_channels=1):
        super().__init__(in_instances, backbone_out_features=1000)

        self.base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(in_features=512, out_features=1000, bias=True)


class ModelDenseNet121_fvs(_BaseModel_fvs):

    def __init__(self, in_instances, in_channels=1):
        super().__init__(in_instances, backbone_out_features=1000)

        self.base_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.base_model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.classifier = nn.Linear(in_features=1024, out_features=1000, bias=True)


if __name__ == '__main__':
    print()
