import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter


# Validation iteration
def _validate(model, loader, loss_function, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, _ in enumerate(loader):
            print(f'[VALIDATE] Batch {i + 1}/{len(loader)}', end='\r')
            images = _[0].to(device)
            labels = _[1].to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(loader)

# Training iteration
def _train(model, loader, optimizer, loss_function, device):
    model.train()
    running_loss = 0.0
    for i, _ in enumerate(loader):
        print(f'[TRAIN] Batch {i + 1}/{len(loader)}', end='\r')
        images = _[0].to(device)
        labels = _[1].to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

class DLPipeline:
    def __init__(self, model, optimizer, loss_function,
                 train_loader, val_loader, name=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = self.model.to(self.device)
        self.best_model = {
            "model": None,
            "loss": float('inf')
        }
        self.name = name
        self.training_losses = []
        self.validation_losses = []
        # Initialize TensorBoard SummaryWriter
        self.tb_writer = SummaryWriter()
        self.save_path = self.tb_writer.log_dir

    def train(self, epochs):
        self.training_losses = []
        self.validation_losses = []
        for epoch in range(epochs):
            print(f'[TRAIN] Epoch {epoch + 1}/{epochs}')
            train_loss = _train(self.model, self.train_loader, self.optimizer, self.loss_function, self.device)
            self.training_losses.append(train_loss)
            print(f'[TRAIN] Epoch {epoch + 1}/{epochs} Loss: {train_loss}')

            val_loss = _validate(self.model, self.val_loader, self.loss_function, self.device)
            self.validation_losses.append(val_loss)
            print(f'[VALIDATE] Epoch {epoch + 1}/{epochs} Loss: {val_loss}')

            if val_loss < self.best_model['loss']:
                self.best_model['model'] = self.model
                self.best_model['loss'] = val_loss
                self.save_model()
                print(f'[SAVE] Best model saved with loss: {val_loss}')

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data in self.val_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                predicted = torch.round(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Generate classification report
        report = classification_report(all_labels, all_predictions, target_names=['Class 0', 'Class 1'])
        return report


    def plot_losses(self):
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.legend()
        plt.show()

    def save_model(self):
        torch.save(self.model, f'{self.save_path}/best_model_{self.name}.pt')