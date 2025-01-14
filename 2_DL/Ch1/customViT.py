# %%
import copy
import datetime
import json
import os
import time
import logging
import random

import numpy as np
import torch
import cv2

from matplotlib import pyplot as plt

from skimage.color import rgb2gray
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# UTILS
class ZScoreNormalize:
    def __call__(self, img):
        mean = torch.mean(img, dim=(1, 2), keepdim=True)  # Compute mean for each channel
        std = torch.std(img, dim=(1, 2), keepdim=True)    # Compute std for each channel
        return (img - mean) / (std + 1e-8)

class ZScoreNormalize:
    def __call__(self, img):
        # Create a mask to ignore zero values
        mask = img != 0

        # Compute mean and std for each channel, considering only non-zero values
        mean = torch.sum(img * mask, dim=(1, 2), keepdim=True) / mask.sum(dim=(1, 2), keepdim=True)
        std = torch.sqrt(torch.sum(((img - mean) ** 2) * mask, dim=(1, 2), keepdim=True) / mask.sum(dim=(1, 2), keepdim=True))

        # Normalize image by subtracting mean and dividing by standard deviation
        return (img - mean) / (std + 1e-8)


def calculate_accuracy(model, data_loader, device):

    model.eval() #model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (100 * correct) / total

    return accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True #What is cudnn?
    torch.backends.cudnn.benchmark = False


# LOADER CLASS
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.samples[index][0]

        return image, label, path


def get_data_loaders(train_dir, val_dir, batch_size, augmentations=[transforms.ToTensor()], seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    # Step 1: Load validation set WITHOUT transforms
    val_dataset_raw = datasets.ImageFolder(val_dir)

    # Step 2: Extract labels for stratified splitting
    val_labels = [label for _, label in val_dataset_raw.samples]
    val_indices = np.arange(len(val_labels))

    # Step 3: Stratified split into validation and test indices
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(stratified_split.split(val_indices, val_labels))

    # Step 4: Define transforms AFTER splitting
    train_transform = transforms.Compose(augmentations)

    valid_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ZScoreNormalize(),
    ])

    # Step 5: Reload datasets with transforms and apply subsets
    train_dataset = ImageFolderWithPaths(train_dir, transform=train_transform)
    val_dataset = ImageFolderWithPaths(val_dir, transform=valid_test_transform)
    test_dataset = ImageFolderWithPaths(val_dir, transform=valid_test_transform)

    # Use Subset to create validation and test sets
    new_val_dataset = Subset(val_dataset, val_idx)
    test_dataset = Subset(test_dataset, test_idx)

    # Step 6: Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(new_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(new_val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

# LOADER SAMPLES
def show_dataloader_samples(dataloader, num_samples=8, randomize=False):
    # Collect one batch of data from the DataLoader
    data_iter = iter(dataloader)
    images, labels, _ = next(data_iter)

    # Select the indices of the samples to display
    indices = list(range(len(images)))
    if randomize:
        indices = random.sample(indices, min(num_samples, len(indices)))
    else:
        indices = indices[:num_samples]

    # Set up the plot grid
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    if len(indices) == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one sample

    for i, idx in enumerate(indices):
        img = images[idx].permute(1, 2, 0).numpy()  # Convert Tensor to NumPy
        label = labels[idx].item() if isinstance(labels[idx], torch.Tensor) else labels[idx]

        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# TRAIN FUNCTION
def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')

    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device,
                save_path="best_model_bin_2.pth", seed=42):
    set_seed(seed)
    model.to(device)

    # Store losses
    train_losses = []
    val_losses = []

    # Track the best validation accuracy and model state
    best_val_accuracy = 0.0
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backprop and optimize
            loss.backward()
            optimizer.step()

            # Calculate Loss
            running_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        # Average loss from all the batches
        epoch_loss = running_loss / len(train_loader.dataset)  # avg loss bc of batches
        train_losses.append(epoch_loss)
        train_accuracy = correct_train / total_train

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        # Average validation loss for epoch
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accuracy = correct_val / total_val

        # Save best model based on val accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            best_model_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_accuracy": val_accuracy
            }

            torch.save(best_model_state, save_path)
            print(f"Best model saved with val acc: {val_accuracy:.4f} at epoch {epoch + 1}")

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        print(f"Epoch {epoch + 1} / {num_epochs}",
              f"Training Loss: {epoch_loss:.4f}",
              f"Training Accuracy: {train_accuracy:.4f}",
              f"Validation Loss: {epoch_val_loss:.4f}",
              f"Validation Accuracy: {val_accuracy:.4f}"
              f"LR: {optimizer.param_groups[0]['lr']:.10f}")

        plot_losses(train_losses, val_losses)

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")

    return model, train_losses, val_losses

import mlflow
import mlflow.pytorch

def train_model_mlflow(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device,
                save_path="best_model_bin_2.pth", seed=42):
    set_seed(seed)
    model.to(device)

    # Store losses
    train_losses = []
    val_losses = []

    # Track the best validation accuracy and model state
    best_val_accuracy = 0.0
    best_model_state = None

    # Start MLflow run
    mlflow.set_experiment("Ch1")
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "optimizer": optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else None
        })

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backprop and optimize
                loss.backward()
                optimizer.step()

                # Calculate Loss
                running_loss += loss.item() * inputs.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            # Average loss from all the batches
            epoch_loss = running_loss / len(train_loader.dataset)  # avg loss bc of batches
            train_losses.append(epoch_loss)
            train_accuracy = correct_train / total_train

            # Validation
            model.eval()
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels, _ in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)

                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            # Average validation loss for epoch
            epoch_val_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            val_accuracy = correct_val / total_val

            # Log metrics
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_accuracy": train_accuracy,
                "val_loss": epoch_val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)

            # Save best model based on val accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                best_model_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_accuracy": val_accuracy
                }

                torch.save(best_model_state, save_path)
                print(f"Best model saved with val acc: {val_accuracy:.4f} at epoch {epoch + 1}")

                mlflow.sklearn.log_model(best_model_state, artifact_path="model",
                                         registered_model_name="HF_ViT"
                                         )

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()

            print(f"Epoch {epoch + 1} / {num_epochs}",
                  f"Training Loss: {epoch_loss:.4f}",
                  f"Training Accuracy: {train_accuracy:.4f}",
                  f"Validation Loss: {epoch_val_loss:.4f}",
                  f"Validation Accuracy: {val_accuracy:.4f}"
                  f"LR: {optimizer.param_groups[0]['lr']:.10f}")

            plot_losses(train_losses, val_losses)

        # Log the best model
        mlflow.pytorch.log_model(model, "best_model")

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")

    return model, train_losses, val_losses

import sys

def train_model_mlflow(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device,
                save_path="best_model_bin_2.pth", seed=42):
    set_seed(seed)
    model.to(device)

    # Store losses
    train_losses = []
    val_losses = []

    # Track the best validation accuracy and model state
    best_val_accuracy = 0.0
    best_model_state = None

    # Start MLflow run
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "optimizer": optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else None
        })

        for epoch in range(num_epochs):
            model.train()
            # Initialize variables for running accuracy and loss
            running_correct = 0
            running_total = 0
            running_loss = 0.0

            for batch_idx, (inputs, labels, _) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device).float()

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backprop and optimize
                loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += loss.item() * inputs.size(0)

                # Calculate predictions and update running accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                running_correct += (preds == labels).sum().item()
                running_total += labels.size(0)

                # Calculate running accuracy
                running_accuracy = running_correct / running_total

                # Update terminal with current batch info
                current_batch = batch_idx + 1
                total_batches = len(train_loader)

                # Print the updated progress for the current batch
                sys.stdout.write(f"\r{current_batch}/{total_batches}\t\t" +
                                 f"current: {(preds == labels).sum().item()}/{preds.__len__()}\t\t" +
                                 f"total: {running_correct}/{running_total} => acc: {running_accuracy:.4f}\t\t" +
                                 f"loss: {running_loss / (current_batch * inputs.size(0)):.4f}")
                sys.stdout.flush()


            # Average loss from all the batches
            epoch_loss = running_loss / len(train_loader.dataset)  # avg loss bc of batches
            train_losses.append(epoch_loss)
            train_accuracy = running_correct / running_total

            # Validation loop
            model.eval()  # Set the model to evaluation mode
            val_running_correct = 0
            val_running_total = 0
            val_running_loss = 0.0

            with torch.no_grad():  # Disable gradient calculation
                for val_batch_idx, (inputs, labels, _) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device).float()

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Update running loss
                    val_running_loss += loss.item() * inputs.size(0)

                    # Calculate predictions and update running accuracy
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_running_correct += (preds == labels).sum().item()
                    val_running_total += labels.size(0)

                    # Calculate running accuracy for validation
                    val_running_accuracy = val_running_correct / val_running_total

                    # Update terminal with current batch info
                    current_val_batch = val_batch_idx + 1
                    total_val_batches = len(val_loader)

                    # Print the updated progress for the current batch
                    sys.stdout.write(f"\rx{current_val_batch}/{total_val_batches}\t\t" +
                                     f"current: {(preds == labels).sum().item()}/{preds.__len__()}\t\t" +
                                     f"hits: {val_running_correct}/{val_running_total} => acc: {val_running_accuracy:.4f}\t\t" +
                                     f"loss: {val_running_loss / (current_val_batch * inputs.size(0)):.4f}")
                    sys.stdout.flush()

            # Average validation loss for epoch
            epoch_val_loss = val_running_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            val_accuracy = val_running_correct / val_running_total

            # Log metrics
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_accuracy": train_accuracy,
                "val_loss": epoch_val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)

            # Save best model based on val accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                best_model_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_accuracy": val_accuracy
                }

                torch.save(best_model_state, save_path)
                print(f"\nBest model saved with val acc: {val_accuracy:.4f} at epoch {epoch + 1}")

                mlflow.sklearn.log_model(best_model_state, artifact_path="model",
                                         registered_model_name="HF_ViT"
                                         )

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()

            print(f"\nEpoch {epoch + 1} / {num_epochs}",
                  f"Training Loss: {epoch_loss:.4f}",
                  f"Training Accuracy: {train_accuracy:.4f}",
                  f"Validation Loss: {epoch_val_loss:.4f}",
                  f"Validation Accuracy: {val_accuracy:.4f}"
                  f"LR: {optimizer.param_groups[0]['lr']:.10f}")

            plot_losses(train_losses, val_losses)

        # Log the best model
        mlflow.pytorch.log_model(model, "best_model")

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")

    return model, train_losses, val_losses



# MODEL CLASS
# MODEL
class SkinLesionClassifier_VIT(nn.Module):
    def __init__(self, num_classes=1, num_unfrozen_layers=8):
        super(SkinLesionClassifier_VIT, self).__init__()

        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the specified number of layers from the end
        for i, layer in enumerate(
                reversed(list(self.model.encoder.layers))):  # Flips order so we start from the final layers
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
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x).squeeze()


from transformers import AutoModelForImageClassification, AutoConfig
class SkinLesionClassifier2_ViT(nn.Module):
    def __init__(self, num_classes=1, num_unfrozen_layers=4):
        super(SkinLesionClassifier2_ViT, self).__init__()

        # Load pretrained model from Hugging Face
        model_name = "Anwarkh1/Skin_Cancer-Image_Classification"
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the specified number of layers from the end
        encoder_layers = list(self.model.vit.encoder.layer)
        for i, layer in enumerate(reversed(encoder_layers)):
            if i < num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                break

        # Replace the classification head
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Get the model output and access the logits
        output = self.model(x)
        return output.logits.squeeze()

# %%
model = SkinLesionClassifier2_ViT(num_classes=1, num_unfrozen_layers=2).to(device)
model

# %%
# LOSS FUNCTION
criterion = torch.nn.BCEWithLogitsLoss()
criterion

# %%
# OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
optimizer

# %%
# LR SCHEDULER
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 3, factor = 0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-6)


# %%
db_pth = r"C:/Users/gimes/Src/repos/CADx-Project/dataset/binary"
train_pth = rf"{db_pth}/train"
val_pth = rf"{db_pth}/val"
test_pth = rf"{db_pth}/test"

# CREATE LOADERS
BATCH_SIZE = 64

augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomResizedCrop(size = 224, scale = (0.8, 1.1)),
        transforms.GaussianBlur(kernel_size = (3,3), sigma = (0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

train_ldr, val_ldr, test_ldr = get_data_loaders(train_pth, val_pth, BATCH_SIZE, augmentations)
show_dataloader_samples(train_ldr)
show_dataloader_samples(val_ldr)

# %%
train_model_mlflow(model, train_ldr, val_ldr, criterion, optimizer, scheduler, num_epochs=20, device=device)
