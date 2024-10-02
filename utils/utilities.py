import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import number
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Loader:
    def __init__(self, path:str, batch_size=32, transform=None):

        # Default transformations (if none provided)
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert image to PyTorch tensor
            ])

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

# STATISTICAL FUNCTIONS FROM DATALOADER
def pixel_level_stats(dataloader: DataLoader):
    # Initialize variables to compute mean and std
    pixel_sum = 0
    pixel_squared_sum = 0
    num_batches = 0
    num_pixels = 0

    # Iterate over the DataLoader to accumulate statistics
    for images, _ in dataloader:
        num_batches += 1
        num_pixels += images.numel() / images.shape[0]  # Number of pixels per image (batch_size, channels, height, width)

        # Compute the sum and squared sum of pixels
        pixel_sum += images.sum()
        pixel_squared_sum += (images ** 2).sum()

    # Compute mean and standard deviation across all images
    mean = pixel_sum / (num_batches * num_pixels)
    std = torch.sqrt((pixel_squared_sum / (num_batches * num_pixels)) - (mean ** 2))

    print(f"Mean: {mean.item()}, Standard Deviation: {std.item()}")


def color_channel_stats(dataloader: DataLoader):
    # Initialize variables to compute mean and std
    num_batches = 0
    num_pixels = 0
    # Initialize variables for RGB channel statistics
    r_sum, g_sum, b_sum = 0, 0, 0
    r_squared_sum, g_squared_sum, b_squared_sum = 0, 0, 0

    for images, _ in dataloader:
        num_batches += 1
        num_pixels += images.numel() / images.shape[0]  # Number of pixels per image (batch_size, channels, height, width)

        r_sum += images[:, 0, :, :].sum()  # Red channel
        g_sum += images[:, 1, :, :].sum()  # Green channel
        b_sum += images[:, 2, :, :].sum()  # Blue channel

        r_squared_sum += (images[:, 0, :, :] ** 2).sum()
        g_squared_sum += (images[:, 1, :, :] ** 2).sum()
        b_squared_sum += (images[:, 2, :, :] ** 2).sum()

    # Compute mean and standard deviation per channel
    r_mean = r_sum / (num_batches * num_pixels / 3)
    g_mean = g_sum / (num_batches * num_pixels / 3)
    b_mean = b_sum / (num_batches * num_pixels / 3)

    r_std = torch.sqrt((r_squared_sum / (num_batches * num_pixels / 3)) - (r_mean ** 2))
    g_std = torch.sqrt((g_squared_sum / (num_batches * num_pixels / 3)) - (g_mean ** 2))
    b_std = torch.sqrt((b_squared_sum / (num_batches * num_pixels / 3)) - (b_mean ** 2))

    print(f"R Mean: {r_mean.item()}, G Mean: {g_mean.item()}, B Mean: {b_mean.item()}")
    print(f"R Std: {r_std.item()}, G Std: {g_std.item()}, B Std: {b_std.item()}")