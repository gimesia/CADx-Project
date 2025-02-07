import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
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
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomResizedCrop(size = 224, scale = (0.8, 1.0)),
        transforms.GaussianBlur(kernel_size = (3,3), sigma = (0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

def get_data_loader_test(test_dir, batch_size, seed = 42):
    g = torch.Generator()
    g.manual_seed(seed)
    

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        #Normalize images with statistis from the pre-trained dataset
        transforms.Normalize( #Normalization operates on tensors and not PIL images
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
        
    ])

    #Load data from folders
    test_dataset = ImageFolderWIthPaths(test_dir, transform = test_transform)

    #Dataloaders for datasets
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False, generator = g)

    return test_loader

