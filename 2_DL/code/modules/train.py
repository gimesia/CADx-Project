# Minimizes loss function by adjusting model parameters through backprop

import torch
import torch.optim as optim
import torch.nn as nn



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):

    """
        Args:
            model: Neural network to train.
            train_loader: DataLoader for training dataset.
            val_loader: DataLoader for validation dataset.
            criterion: Loss function.
            optimizer: Optimization algorithm
            device: Device to use ('cuda' or 'cpu').
            epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
        Returns:
            Trained model.
    """
        
    model.to(device)
    
    #Store losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels, _  in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            #Zero the parameter gradients
            optimizer.zero_grad()
            #Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #Backprop and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)


        #Average loss from all the batches
        epoch_loss = running_loss/len(train_loader.dataset) # avg loss bc of batches
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

        
        #Average validation loss for epoch
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch + 1} / {num_epochs}",
              f"Training Loss: {epoch_loss:.4f}",
              f"Validation Loss: {epoch_val_loss:.4f}")
        

    return model, train_losses, val_losses
