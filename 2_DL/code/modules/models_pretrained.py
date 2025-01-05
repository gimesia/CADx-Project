import torch
import torch.nn as nn
import torchvision.models as models


class SkinLesionClassifier(nn.Module):

    def __init__(self, num_classes = 1):
        super(SkinLesionClassifier, self).__init__()# this class inherits from nn.Module

        self.model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)

        #Replace final FC layer
        #self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.model.fc = nn.Sequential(

            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x).squeeze() #Ensure output is [batch_size]