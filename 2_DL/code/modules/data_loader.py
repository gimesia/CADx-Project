import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


class ImageFolderWIthPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        image, label = super(ImageFolderWIthPaths, self).__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


def get_data_loaders(train_dir, val_dir, batch_size, seed=42):
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
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 5: Reload datasets with transforms and apply subsets
    train_dataset = ImageFolderWIthPaths(train_dir, transform=train_transform)
    val_dataset = ImageFolderWIthPaths(val_dir, transform=valid_test_transform)
    test_dataset = ImageFolderWIthPaths(val_dir, transform=valid_test_transform)

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


def get_data_loader_test(test_dir, batch_size, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    test_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load data from folder
    test_dataset = ImageFolderWIthPaths(test_dir, transform=test_transform)

    # Dataloader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=g)

    return test_loader
