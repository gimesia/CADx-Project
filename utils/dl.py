import datetime
import json
import os
import time
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from torch.utils.data.dataloader import DataLoader

from utils.loader import FactoryLoader
from utils.preprocessing import PreprocessingFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Validation iteration
def _validate(model, loader, loss_function, device, verbose=False):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i % 10 == 0:
                logger.info(f'[VALIDATE] Batch {i + 1}/{len(loader)}')
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)

            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            outputs_ = outputs.cpu().detach().numpy()
            labels_ = labels.cpu().detach().numpy()

            outputs_ = np.round(outputs_, decimals=0)

            if verbose:
                print(f"Labels: {labels_.squeeze()}")
                print(f"Predictions: {outputs_.squeeze()}")
                print(f"Hits: {(np.sum(outputs_ * labels_).astype(int))}/{len(labels_)}")
                print(f"Loss: {loss.item()}")

    return val_loss / len(loader)

# Training iteration
def _train(model, loader, optimizer, loss_function, device, verbose=False):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(loader):
        if i % 10 == 0:
           logger.info(f'[TRAIN] Batch {i + 1}/{len(loader)}')
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)

        outputs_ = outputs.cpu().detach().numpy()
        labels_ = labels.cpu().detach().numpy()

        outputs_ = np.round(outputs_, decimals=0)

        if verbose:
            print(f"Labels: {labels_.squeeze()}")
            print(f"Predictions: {outputs_.squeeze()}")
            print(f"Hits: {np.sum(outputs_ * labels_).astype(int)}/{len(labels_)}")
            print(f"Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

class DLPipeline:
    def __init__(self, model, optimizer, loss_function,
                 train_loader: FactoryLoader, val_loader: FactoryLoader, name=None,
                 challange="Ch1"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader.get_loader()
        self.val_loader = val_loader.get_loader()
        self.model = self.model.to(self.device)
        self.best_model = {
            "model": None,
            "loss": float('inf')
        }
        self.name = name
        self.training_losses = []
        self.validation_losses = []
        self.preprocessing_steps = train_loader.get_transformation_steps()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
        self.save_path = os.path.join(project_root, "2_DL", challange, "runs",
                                      datetime.datetime.now().strftime("%m-%d-%Hh%M") + f"_{name}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train(self, epochs):
        self.training_losses = []
        self.validation_losses = []
        for epoch in range(epochs):
            logger.info(f'[EPOCH] {epoch + 1}/{epochs}')
            train_loss = _train(self.model, self.train_loader, self.optimizer, self.loss_function, self.device)
            self.training_losses.append(train_loss)
            logger.info(f'[TRAIN] Epoch {epoch + 1}/{epochs} Loss: {train_loss}')

            val_loss = _validate(self.model, self.val_loader, self.loss_function, self.device)
            self.validation_losses.append(val_loss)
            logger.info(f'[VALIDATE] Epoch {epoch + 1}/{epochs} Loss: {val_loss}')

            if val_loss < self.best_model['loss']:
                self.best_model['model'] = self.model
                self.best_model['loss'] = val_loss
                self.save_model()
                logger.info(f'[SAVE] Best model saved with loss: {val_loss}')

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

    def plot_losses(self, save=True):
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if save:
            plt.savefig(os.path.join(self.save_path, 'losses.png'))
        plt.show()

    def plot_roc_auc(self, save=True):
        self.model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data in self.val_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())

        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        if save:
            plt.savefig(os.path.join(self.save_path, 'roc_auc.png'))
        plt.show()

    def save_hyperparameters(self):
        hyperparameters = {
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "batch_size": self.train_loader.batch_size,
            "epochs": len(self.training_losses),
            "model_name": self.name,
            "preprocessing_steps": self.preprocessing_steps
        }
        with open(os.path.join(self.save_path, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparameters, f, indent=4)

    def save_model(self):
        torch.save(self.model, f'{self.save_path}/best_model.pt')
        self.save_hyperparameters()