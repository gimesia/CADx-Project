import torch.nn as nn
import torchvision.models as models


class SkinLesionClassifier_EfficientNetB7(nn.Module):
    def __init__(self, num_classes=1, num_unfrozen_layers=3):
        super(SkinLesionClassifier_EfficientNetB7, self).__init__()

        # Load a pretrained EfficientNet-B7 model
        self.model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the specified number of layers from the end of the backbone
        layers_to_unfreeze = []
        for name, module in self.model.features.named_children():
            layers_to_unfreeze.append(module)

        # Unfreeze the last `num_unfrozen_layers`
        for module in layers_to_unfreeze[-num_unfrozen_layers:]:
            for param in module.parameters():
                param.requires_grad = True

        # Replace the classifier head for binary classification
        in_features = self.model.classifier[1].in_features  # Access the final layer's input features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x).squeeze() 