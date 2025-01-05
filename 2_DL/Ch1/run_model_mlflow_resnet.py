# %%
# Assuming you have a PreprocessMelanoma factory instance
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from utils.dl_mlflow import MLFlowDLPipeline
from utils.loader import FactoryLoader
from utils.preprocessing import PreprocessingFactory

from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet152, ResNet152_Weights,
    resnet101, ResNet101_Weights,
    wide_resnet101_2, Wide_ResNet101_2_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    vgg19_bn, VGG19_BN_Weights,
    convnext_base, ConvNeXt_Base_Weights
)

# Dataset path
TRAIN_PATH = r"C:\Users\gimes\Src\repos\CADx-Project\dataset\binary\train"
VAL_PATH = r"C:\Users\gimes\Src\repos\CADx-Project\dataset\binary\val"


# Create preprocessing factory
factory = PreprocessingFactory()
factory.pad2square(fill=0)  # Pad images to form square shape, filling with 0
factory.zscore_normalize()
factory.resize((224, 224))

# Create data loaders
PERCENT = 100
# for i in [24,32,48]:
# print("CURRENT BATCH SIZE: ", i)
# print("-----------------------------------")
BATCH_SIZE = 32
train_loader = FactoryLoader(TRAIN_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)
val_loader = FactoryLoader(VAL_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)

# %%
# Create model, optimizer and loss function
model = resnet101(weights=ResNet101_Weights)
model

# %%
# print(model.classifier)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True
model

# print(model.classifier)
model

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Create pipeline
dl = MLFlowDLPipeline(model, optimizer, criterion, train_loader, val_loader, "ResNet101", logits=True)


# val.show_images(80)
# Train pipeline
dl.train(epochs=20)

# Evaluate pipeline
print(dl.evaluate())

# Plot losses
dl.plot_losses()