import torch
import torch.nn as nn
import torchvision.models as models


class SkinLesionClassifier(nn.Module):

    def __init__(self, num_classes = 2):
        super(SkinLesionClassifier, self).__init__()# this class inherits from nn.Module

        self.model = models.resnet18(pretrained=True)

        #Replace final FC layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    
    def forward(self, x):
        return self.model(x)