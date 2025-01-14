# %%
# Assuming you have a PreprocessMelanoma factory instance
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils.ModelWrapper import ModelWrapper
from utils.dl_mlflow import MLFlowDLPipeline
from utils.loader import FactoryLoader
from utils.preprocessing import PreprocessingFactory

from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet152, ResNet152_Weights,
    resnet101, ResNet101_Weights,
    # wide_resnet101_2, Wide_ResNet101_2_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    # vgg19_bn, VGG19_BN_Weights,
    # convnext_base, ConvNeXt_Base_Weights
)

# Dataset path
TRAIN_PATH = r"/dataset/binary/train"
VAL_PATH = r"/dataset/binary/val"


# Create preprocessing factory
factory = PreprocessingFactory()
factory.pad2square(fill=0)  # Pad images to form square shape, filling with 0
factory.zscore_normalize()
factory.resize((380, 380))

# Create data loaders
PERCENT = 100
BATCH_SIZE = 64
train_loader = FactoryLoader(TRAIN_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)
val_loader = FactoryLoader(VAL_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)

# %%
# Create model, optimizer and loss function

class EffNetMv2(nn.Module):
    def __init__(self, num_classes=1, trained_layers=3, dropout=0.3):
        super(EffNetMv2, self).__init__()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)

        num_features = self.model.classifier[1].in_features

        # print("Old classifier: ", self.model.classifier)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

        # print("New classifier: ", self.model.classifier)

        # Unfreeze the last n layers
        for i, layer in enumerate(list(self.model.features)[::-1]):
            print(i)
            if i == trained_layers:
                break
            print("Unfreezing layer: ", layer)
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_data):
        return self.model(input_data)


for layers in range(1, 6, 2):
    model = EffNetMv2(num_classes=1, trained_layers=0)

    # %%
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2)
    criterion = nn.BCELoss()

    # Create pipeline
    dl = MLFlowDLPipeline(model, optimizer, criterion, train_loader, val_loader,
                          f"{model.__class__.__name__}_{layers}-trained-layers", challenge="Ch1")

    # val.show_images(80)
    # Train pipeline
    dl.train(epochs=25)

    # Evaluate pipeline
    print(dl.evaluate())

    # Plot losses
    dl.plot_losses()