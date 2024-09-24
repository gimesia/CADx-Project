import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class BinaryLoader:
    def __init__(self, path:str, batch_size=32, transform=None):

        # Default transformations (if none provided)
        if transform is None:
            transform = transforms.Compose([
                # transforms.Resize((224, 224)),  # Resize to 224x224
                transforms.ToTensor(),  # Convert image to PyTorch tensor
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])

        self.dataset = datasets.ImageFolder(path, transform=transform)
        self.batch_size = batch_size

    def get_loader(self, shuffle=True):
        # Create DataLoader
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=shuffle)

    def get_num_classes(self):
        # Return number of classes (should be 2 for binary classification)
        return len(self.dataset.classes)


class MulticlassImageLoader:
    def __init__(self, path, batch_size=32, transform=None):
        # Default transformations (if none provided)
        if transform is None:
            transform = transforms.Compose([
                # transforms.Resize((224, 224)),  # Resize to 224x224
                transforms.ToTensor(),  # Convert image to PyTorch tensor
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])
        # Create dataset using ImageFolder
        self.dataset = datasets.ImageFolder(root=path, transform=transform)
        self.batch_size = batch_size

    def get_loader(self, shuffle=False):
        # Create DataLoader
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=shuffle)

    def get_num_classes(self):
        # Return number of classes (e.g., 3 for 3-class classification)
        return len(self.dataset.classes)

