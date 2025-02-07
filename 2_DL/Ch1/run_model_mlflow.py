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
factory.pad2square(fill=np.nan)  # Pad images to form square shape, filling with 0
factory.normalize2float()
factory.resize((224, 224))

# Create data loaders
PERCENT = 100
for i in [24,32,48]:
    print("CURRENT BATCH SIZE: ", i)
    print("-----------------------------------")
    BATCH_SIZE = i
    train_loader = FactoryLoader(TRAIN_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)
    val_loader = FactoryLoader(VAL_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)

    # %%
    # Create model, optimizer and loss function
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)
    # Freeze the feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    model

    # %%
    # print(model.classifier)
    num_features = model.classifier[1].in_features  # Get the number of input features to the classifier
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    # print(model.classifier)
    model.classifier
    # %%
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Create pipeline
    dl = MLFlowDLPipeline(model, optimizer, criterion, train_loader, val_loader, "EfficientNet_V2_M")


    # val.show_images(80)
    # Train pipeline
    dl.train(epochs=50)

    # Evaluate pipeline
    print(dl.evaluate())

    # Plot losses
    dl.plot_losses()