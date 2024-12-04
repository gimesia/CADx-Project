import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset


class ImageFolderWIthPaths(datasets.ImageFolder):

    def __getitem__(self, index):

        image, label = super(ImageFolderWIthPaths, self).__getitem__(index)

        path = self.samples[index][0]

        return image, label, path

def get_data_loaders(train_dir, val_dir, batch_size, subset_percentage = 1.0, seed = 42):
    
    g = torch.Generator()
    g.manual_seed(seed)

    train_transform = transforms.Compose([


        
        #LOOK FOR THE BEST AUGMENTATIONS TO USE
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast = 0.1, saturation = 0.1),
        transforms.Resize((224,224)),
        #transforms.RandomResizedCrop(224), #Up might limit the effect of subsequent spatial trans
        transforms.ToTensor(),
        transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )

    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        #Normalize images with statistis from the pre-trained dataset
        transforms.Normalize( #Normalization operates on tensors and not PIL images
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
        
    ])

    #Load data from folders
    train_dataset = ImageFolderWIthPaths(train_dir, transform = train_transform)
    val_dataset = ImageFolderWIthPaths(val_dir, transform = valid_transform)

    #Dataloaders for datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)

    return train_loader, val_loader

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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True, generator = g)

    return test_loader

