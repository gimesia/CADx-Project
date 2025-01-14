# %%
# Assuming you have a PreprocessMelanoma factory instance
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils.ModelWrapper import ModelWrapper
from utils.dl_mlflow import MLFlowDLPipeline
from utils.loader import FactoryLoader
from utils.preprocessing import PreprocessingFactory

from torchvision.models import (
    resnet50, ResNet50_Weights,
    # resnet152, ResNet152_Weights,
    # resnet101, ResNet101_Weights,
    wide_resnet101_2, Wide_ResNet101_2_Weights,
    # efficientnet_v2_m, EfficientNet_V2_M_Weights,
    # vgg19_bn, VGG19_BN_Weights,
    # convnext_base, ConvNeXt_Base_Weights
)

# Dataset path
TRAIN_PATH = r"/dataset/binary/train"
VAL_PATH = r"/dataset/binary/val"


# Create preprocessing factory
factory = PreprocessingFactory()
factory.pad2square(fill=0)  # Pad images to form square shape, filling with 0
factory.normalize2float()
factory.resize((224, 224))

# Create data loaders
PERCENT = 100
BATCH_SIZE = 64
train_loader = FactoryLoader(TRAIN_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)
val_loader = FactoryLoader(VAL_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)

# %%
# Create model, optimizer and loss function

class ResNet50(nn.Module):
    def __init__(self, num_classes=1, freeze_layers=False, trained_layers=3, dropout=0.3):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights)


        if freeze_layers:
            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            print(self.model.named_parameters)
            # Unfreeze the last n layers
            correction = 0
            for i, layer in enumerate(list(self.model.named_children())[::-1]):
                name, layer = layer
                if name in ["fc", "avgpool", "maxpool", "relu"]:
                    correction += 1
                    continue
                if i == trained_layers+correction:
                    break
                print(f"Unfreezing {name}: ", )#layer)
                for param in layer.parameters():
                    param.requires_grad = True

        # Classifier
        num_features = self.model.fc.in_features
        # print("Old classifier: ", self.model.fc)

        self.model.fc = torch.nn.Linear(num_features, 1)
        self.model.fc =  nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        # print("New classifier: ", self.model.fc)



    def trainable_parameters(self):
        return len([param for param in self.model.parameters() if param.requires_grad])

    def forward(self, input_data):
        return self.model(input_data)


# %%
model = ResNet50(num_classes=1, freeze_layers=False, trained_layers=1)
model.trainable_parameters()


# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=2)
criterion = nn.BCELoss()

# Create pipeline
dl = MLFlowDLPipeline(model, optimizer, criterion, train_loader, val_loader,
                      name=f"{model.__class__.__name__}", lr_scheduler=lr_scheduler, challenge="Ch1")


# val.show_images(80)
# Train pipeline
dl.train(epochs=40)

# Evaluate pipeline
print(dl.evaluate())

# Plot losses
dl.plot_losses()