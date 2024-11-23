import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(train_dir, val_dir, batch_size):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()

    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        
    ])

    #Load data from folders
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform = test_transform)

    #Dataloaders for datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)

    return train_loader, val_loader

