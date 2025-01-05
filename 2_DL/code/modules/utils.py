import torch
import matplotlib.pyplot as plt
import random
import numpy as np

def calculate_accuracy(model, data_loader, device):

    model.eval() #model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (100 * correct) / total

    return accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True #What is cudnn?
    torch.backends.cudnn.benchmark = False