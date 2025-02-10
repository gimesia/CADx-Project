import torch
import torch.nn as nn
import torchvision.models as models


class SkinLesionClassifier(nn.Module):

    def __init__(self, num_classes = 1):
        super(SkinLesionClassifier, self).__init__()# this class inherits from nn.Module

        self.model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)

        #Most vanilla one
        #Replace final FC layer
        #self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        #Number of features in the last FC layer of ResNet

        in_features = self.model.fc.in_features # This is 2048


        self.model.fc = nn.Sequential(
            #Some FC layers added
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 


            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)


        )
    
    def forward(self, x):
        return self.model(x).squeeze() #Ensure output is [batch_size]