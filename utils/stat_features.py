import torch
from torch.utils.data import DataLoader


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