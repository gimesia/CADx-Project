import roch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self, input_channels = 3, num_classes = 2):
        super(BinaryClassifier, self).__init__() #To properly initialize the parent class nn.Module
        
        self.conv_layers = nn.Sequential(
            nn.conv2d(input_channels, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2), #Reduction of spatial dim by 2
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2) #Further reduces spatial dim by 2

        )

        #Adapative poolin to generated fixed spatial size (1x1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1)) #Reduces feat maps to 1x1

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor: Logits of shape (batch_size, num_classes)
        """

        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = self.fc_layers(x)

        return x 
        
