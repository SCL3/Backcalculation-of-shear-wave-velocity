import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DL Architecture
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

if __name__ == '__main__':
    print()