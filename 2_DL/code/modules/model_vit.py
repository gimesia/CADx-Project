import torch.nn as nn
import torchvision.models as models

class SkinLesionClassifier_VIT(nn.Module):
    def __init__(self, num_classes=1, num_unfrozen_layers=8):
        super(SkinLesionClassifier_VIT, self).__init__()

        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the specified number of layers from the end
        for i, layer in enumerate(reversed(list(self.model.encoder.layers))): #Flips order so we start from the final layers
            if i < num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                break

        # Replace the classification head

        in_features = self.model.heads.head.in_features
        self.model.heads = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace = True),
            nn.Dropout(0.6),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),

            nn.Linear(512, num_classes)



        )

    def forward(self, x):
        return self.model(x).squeeze()
