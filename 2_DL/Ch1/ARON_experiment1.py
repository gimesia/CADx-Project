# Assuming you have a PreprocessMelanoma factory instance
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from utils.dl_mlflow import MLFlowDLPipeline
from utils.loader import FactoryLoader
from utils.preprocessing import PreprocessingFactory


# Create EfficientNet model
class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(EfficientNetBinaryClassifier, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.fc = nn.Linear(self.efficientnet._fc.in_features, 1)
        self.efficientnet._fc = nn.Identity()  # Remove the original fully connected layer

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# Dataset path
TRAIN_PATH = r"C:\Users\gimes\Src\repos\CADx-Project\dataset\binary\train"
VAL_PATH = r"C:\Users\gimes\Src\repos\CADx-Project\dataset\binary\val"


# Create preprocessing factory
factory = PreprocessingFactory()
factory.pad2square(fill=np.nan)  # Pad images to form square shape, filling with 0
factory.normalize2float()
factory.resize((224, 224))

# Create data loaders
PERCENT = 3
BATCH_SIZE = 32
train_loader = FactoryLoader(TRAIN_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)
val_loader = FactoryLoader(VAL_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)

# Create model, optimizer and loss function
model = EfficientNetBinaryClassifier()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Create pipeline
dl = MLFlowDLPipeline(model, optimizer, criterion, train_loader, val_loader, "CustomCNNmlflow")


# val.show_images(80)
# Train pipeline
dl.train(epochs=3)

# Evaluate pipeline
print(dl.evaluate())

# Plot losses
dl.plot_losses()