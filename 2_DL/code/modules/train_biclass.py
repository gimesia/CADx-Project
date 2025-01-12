import torch
import torch.optim as optim
import torch.nn as nn
import os
from modules.utils import set_seed


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path="best_model_multiclass.pth", seed=42):
    """
        Args:
            model: Neural network to train.
            train_loader: DataLoader for training dataset.
            val_loader: DataLoader for validation dataset.
            criterion: Loss function.
            optimizer: Optimization algorithm
            device: Device to use ('cuda' or 'cpu').
            num_epochs: Number of training epochs.
            save_path: Path to save the best model.
            seed: Random seed for reproducibility.
        Returns:
            Trained model, training losses, and validation losses.
    """
    set_seed(seed)

    model.to(device)
    
    # Store losses
    train_losses = []
    val_losses = []

    # Track best validation accuracy and model state
    best_val_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()  # Labels must be integers for CrossEntropyLoss

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Calculate loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        # Average training loss
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracy = correct_train / total_train

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs, dim=1)  # Get predicted class
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        # Average validation loss and accuracy
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accuracy = correct_val / total_val

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            best_model_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_accuracy": val_accuracy
            }
            
            torch.save(best_model_state, save_path)
            print(f"Best model saved with val acc: {val_accuracy:.4f} at epoch {epoch + 1}")

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        print(f"Epoch {epoch + 1} / {num_epochs}",
              f"Training Loss: {epoch_loss:.4f}",
              f"Training Accuracy: {train_accuracy:.4f}",
              f"Validation Loss: {epoch_val_loss:.4f}",
              f"Validation Accuracy: {val_accuracy:.4f}",
              f"LR: {optimizer.param_groups[0]['lr']:.10f}")

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")

    return model, train_losses, val_losses
