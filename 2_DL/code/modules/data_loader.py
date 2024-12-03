import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset


class ImageFolderWIthPaths(datasets.ImageFolder):

    def __getitem__(self, index):

        image, label = super(ImageFolderWIthPaths, self).__getitem__(index)

        path = self.samples[index][0]

        return image, label, path


def get_data_loaders(train_dir, val_dir, batch_size, subset_percentage = 1.0):
    

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
    train_dataset = ImageFolderWIthPaths(train_dir, transform = train_transform)
    val_dataset = ImageFolderWIthPaths(val_dir, transform = test_transform)

    #Dataloaders for datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)

    return train_loader, val_loader

