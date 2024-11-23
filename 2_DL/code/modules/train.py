# Minimizes loss function by adjusting model parameters through backprop

import torch
import torch.optim as optim
import torch.nn as nn



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):

    """
        Args:
            model: Neural network to train.
            train_loader: DataLoader for training dataset.
            test_loader: DataLoader for testing dataset.
            criterion: Loss function.
            optimizer: Optimization algorithm
            device: Device to use ('cuda' or 'cpu').
            epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
        Returns:
            Trained model.
    """
        


    model.to(device)
    

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            #Zero the parameter gradients

            optimizer.zero_grad()
            #Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #Backprop and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)


            #Continue with validation here
        epoch_loss = running_loss/len(train_loader.dataset) # avg loss bc of batches

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}") # Average loss across all batches

    return model
