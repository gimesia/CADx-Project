import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision

def denormalize(img, mean, std):

    mean = np.array(mean)
    std = np.array(std)

    mean = mean.reshape(1, 1, 1 , 3)
    std = std.reshape(1, 1, 1, 3)
    
    #Denormalized
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def visualize_dataloader(dataloader, mean, std, num_images = 5):
    """mean used for normalization
        std used for normalization
    """

    #Get batch of images
    dataiter = iter(dataloader)
    images, labels, paths = next(dataiter)

    print(f"Type of images: {type(images)}")
    print(f"Type of labels: {type(labels)}")    
    print(f"Type of paths: {type(paths)}")



    num_images = int(num_images)
    images = images[:num_images]
    labels = labels[:num_images]
    paths = paths[:num_images]



    images = images.numpy()
    images = np.transpose(images, (0, 2, 3, 1))

    images = denormalize(images, mean, std)

    class_names = dataloader.dataset.classes

    for idx in range(num_images):
        plt.figure(figsize=(4,4))
        plt.imshow(images[idx])
        label_idx = labels[idx].item()
        class_name = class_names[label_idx]
        file_name = os.path.basename(paths[idx])
        plt.title(f'Label: {class_name} ({label_idx}) \nFile:{file_name}')
        plt.axis('off')
        plt.show()



