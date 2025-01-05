# %%
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from utils.dl_mlflow import MLFlowDLPipeline
from utils.loader import FactoryLoader
from utils.preprocessing import PreprocessingFactory

from transformers import ViTForImageClassification, ViTFeatureExtractor, AutoModelForImageClassification

# %%
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

BATCH_SIZE = 48
train_loader = FactoryLoader(TRAIN_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)
val_loader = FactoryLoader(VAL_PATH, batch_size=BATCH_SIZE, factory=factory, percentage=PERCENT, shuffle=True)

# %%
model_name = "Anwarkh1/Skin_Cancer-Image_Classification"
model = AutoModelForImageClassification.from_pretrained(model_name)

# Freeze the feature extractor
for param in model.parameters():
    param.requires_grad = False

model

# %%
# print(model.classifier)
num_features = model.classifier.in_features  # Get the number of input features to the classifier
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1),
    # nn.Sigmoid()
)
# print(model.classifier)
model
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Create pipeline
dl = MLFlowDLPipeline(model, optimizer, criterion, train_loader, val_loader, "ViT_Anwarkh1")


# val.show_images(80)
# Train pipeline
dl.train(epochs=20)

# Evaluate pipeline
print(dl.evaluate())

# Plot losses
dl.plot_losses()