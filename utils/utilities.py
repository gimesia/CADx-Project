import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.preprocessing import PreprocessingFactory, PreprocessMelanoma


class Loader:
    def __init__(self, path: str, batch_size=32, transform=None):

        # Default transformations (if none provided)
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert image to PyTorch tensor
            ])

        # Load the dataset using the transformation pipeline
        self.dataset = datasets.ImageFolder(path, transform=transform)
        self.batch_size = batch_size
        self.__instance = None

    def get_loader(self, shuffle=False) -> DataLoader:
        # Create DataLoader
        if self.__instance is None:
            self.__instance = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=shuffle)
        return self.__instance

    def get_num_classes(self) -> int:
        # Return number of classes (should be 2 for binary classification, n for multiclass)
        return len(self.dataset.classes)

    def show_images(self, num_images=8, randomize=False):
        # Determine number of rows based on number of images and fixed 4 columns
        num_columns = 4
        num_rows = (num_images + num_columns - 1) // num_columns  # Ceiling division for grid rows

        loader = self.get_loader()  # Get the DataLoader

        images_list = []
        labels_list = []

        # Continue loading batches until we have enough images
        for batch in loader:
            images, labels = batch

            # Convert to numpy for plotting (assuming CHW format)
            if torch.is_tensor(images):
                images = images.permute(0, 2, 3, 1).numpy()  # Change to (batch_size, height, width, channels) if needed

            images_list.append(images)
            labels_list.append(labels)

            # If we have enough images, break the loop
            if sum(len(imgs) for imgs in images_list) >= num_images:
                break

        # Concatenate all loaded images and labels
        all_images = np.concatenate(images_list, axis=0)
        all_labels = torch.cat(labels_list, dim=0) if torch.is_tensor(labels_list[0]) else np.concatenate(labels_list)

        # Select a subset of images, random or not
        indices = list(range(len(all_images)))
        if randomize:
            random.shuffle(indices)

        selected_indices = indices[:num_images]

        # Create the plot
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
        axs = axs.flatten()  # Flatten to easily iterate over all subplots

        for i, idx in enumerate(selected_indices):
            image = all_images[idx]
            label = all_labels[idx].item() if torch.is_tensor(all_labels[idx]) else all_labels[idx]

            # Plot the image
            axs[i].imshow(np.squeeze(image))  # Squeeze for grayscale images
            axs[i].set_title(f"idx: {idx}; label: {label}")
            axs[i].axis('off')  # Hide axis

        # Hide any unused subplots
        for i in range(num_images, len(axs)):
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()


class FactoryLoader:
    def __init__(self, path: str, batch_size=32, factory: PreprocessingFactory = None):

        # Define the transformation pipeline
        if factory is not None:
            transform = transforms.Compose([
                transforms.Lambda(lambda img: np.array(img)),  # Convert PIL to NumPy
                PreprocessMelanoma(factory),  # Apply the factory-based preprocessing
            ])
        else:
            # Default transformation if no factory is provided
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert image to PyTorch tensor
            ])

        # Load the dataset using the transformation pipeline
        self.batch_size = batch_size
        self.__factory = factory
        self.__dataset = datasets.ImageFolder(path, transform=transform)
        self.__instance = None

    def get_loader(self, shuffle=False) -> DataLoader:
        # Create DataLoader
        if self.__instance is None:
            self.__instance = DataLoader(dataset=self.__dataset, batch_size=self.batch_size, shuffle=shuffle)
        return self.__instance

    def get_num_classes(self) -> int:
        return len(self.__dataset.classes)

    def get_classes(self) -> list:
        return self.__dataset.classes

    def get_transformation_steps(self):
        return self.__factory.get_steps()

    def show_images(self, num_images=8, randomize=False):
        # Determine number of rows and columns for the grid
        num_columns = 4
        num_rows = (num_images + num_columns - 1) // num_columns

        loader = self.get_loader()

        images_list = []
        labels_list = []

        for batch in loader:
            images, labels = batch
            images = images.permute(0, 2, 3, 1).numpy()

            images_list.append(images)
            labels_list.append(labels)

            if sum(len(imgs) for imgs in images_list) >= num_images:
                break

        all_images = np.concatenate(images_list, axis=0)
        all_labels = torch.cat(labels_list, dim=0)

        indices = list(range(len(all_images)))
        if randomize:
            random.shuffle(indices)

        selected_indices = indices[:num_images]

        fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
        axs = axs.flatten()

        for i, idx in enumerate(selected_indices):
            image = all_images[idx]
            label = all_labels[idx].item()

            axs[i].imshow(np.squeeze(image))
            axs[i].set_title(f"idx: {idx}; label: {label}")
            axs[i].axis('off')

        for i in range(num_images, len(axs)):
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()
