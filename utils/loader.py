import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
from utils.preprocessing import PreprocessingFactory, PreprocessMelanoma

# To create a balanced subset
def balanced_subset(dataset, percentage=100):
    # If percentage is 100, use all dataset indices
    if percentage == 100:
        return list(range(len(dataset)))

    # Get the class indices
    class_indices = {cls_idx: np.where(np.array(dataset.targets) == cls_idx)[0]
                     for cls_idx in range(len(dataset.classes))}

    # Calculate number of images to load based on percentage
    total_images = int(len(dataset) * (percentage / 100.0))
    per_class_images = max(1, total_images // len(dataset.classes))

    # Sample equal number of images per class
    subset_indices = []
    for cls_idx, indices in class_indices.items():
        subset_indices.extend(indices[:per_class_images])

    return subset_indices

class FactoryLoader:
    def __init__(self, path: str, batch_size=32,
                 factory: PreprocessingFactory = None, percentage=100, shuffle=False,
                 augmentations=None):

        # Define the transformation pipeline
        transform_list = []

        # Add augmentations if provided
        if augmentations:
            transform_list.extend(augmentations)

        # Add factory-based preprocessing or default preprocessing
        if factory is not None:
            transform_list.extend([
                transforms.Lambda(lambda img: np.array(img)),  # Convert PIL to NumPy
                PreprocessMelanoma(factory),  # Apply the factory-based preprocessing
            ])
        else:
            transform_list.append(transforms.ToTensor())  # Convert image to PyTorch tensor

        transform = transforms.Compose(transform_list)

        # Load the dataset using the transformation pipeline
        self.batch_size = batch_size
        self.__factory = factory
        self.random = shuffle
        dataset = datasets.ImageFolder(path, transform=transform)

        # Get a balanced subset of the dataset
        subset_indices = balanced_subset(dataset, percentage=percentage)

        # Create the Subset with balanced data
        self.__dataset = Subset(dataset, subset_indices)

        self.__instance = None

    def get_loader(self) -> DataLoader:
        # Create DataLoader
        if self.__instance is None:
            self.__instance = DataLoader(dataset=self.__dataset,
                                         batch_size=self.batch_size,
                                         shuffle=self.random)
        return self.__instance

    def get_num_classes(self) -> int:
        return len(self.__dataset.dataset.classes)

    def get_classes(self) -> list:
        return self.__dataset.dataset.classes

    def get_size(self) -> list:
        return self.__dataset.__len__()

    def get_transformation_steps(self):
        return self.__factory.get_steps()

    def __len__(self):
        return len(self.__dataset)

    def show_images(self, num_images=8):
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

        indices = np.arange(all_images.shape[0])

        selected_indices = indices[:num_images]

        fig, axs = plt.subplots(num_rows,
                                num_columns,
                                figsize=(15, 5 * num_rows))
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

    def get_element_by_id(self, idx: int):
        if idx < 0 or idx >= len(self.__dataset):
            raise IndexError(f"ID {idx} is out of bounds "
                             f"for dataset with size {len(self.__dataset)}")

        # Get image and label at the specific index
        image, label = self.__dataset[idx]

        # Change shape to rgb (C,H,W) -> (H,W,C)
        image = np.dstack([image[0], image[1], image[2]])
        return image, label

    def get_factory(self):
        return self.__factory

    def change_factory(self, factory: PreprocessingFactory):
        self.__factory = factory
