import copy
import datetime
import json
import os
import time
import logging

import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from openpyxl.styles.builtins import output
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix

from utils.loader import FactoryLoader


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        # Ensure this method matches your model's prediction interface
        return self.model(input_data)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VERBOSE = True

# Validation iteration
def _validate(model, loader, loss_function, device, logits=False, verbose=VERBOSE):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i % 10 == 0:
                logger.info(f'[VALIDATE] Batch {i + 1}/{len(loader)}')
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            if logits:
                outputs = torch.sigmoid(outputs)

            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            if verbose:
                outputs_ = outputs.cpu().detach().numpy()
                outputs_ = np.round(outputs_, decimals=0)
                labels_ = labels.cpu().detach().numpy()

                print(f"Labels: {labels_.squeeze()}")
                print(f"Predictions: {outputs_.squeeze()}")

                print(f"Hits: {(np.sum(outputs_ == labels_).astype(int))}/{len(labels_)}")
                print(f"Loss: {loss.item()}")
            all_labels.extend(list(labels_))
            all_preds.extend(list(outputs_))

    return val_loss / len(loader), classification_report(all_labels, all_preds, output_dict=True)


# Training iteration
def _train(model, loader, optimizer, loss_function, device, logits=False, verbose=VERBOSE):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(loader):
        if i % 10 == 0:
           logger.info(f'[TRAIN] Batch {i + 1}/{len(loader)}')
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        if logits:
            outputs = torch.sigmoid(outputs)

        loss = loss_function(outputs, labels)

        if verbose:
            outputs_ = outputs.cpu().detach().numpy()
            outputs_ = np.round(outputs_, decimals=0)
            labels_ = labels.cpu().detach().numpy()

            print(f"Labels: {labels_.squeeze()}")
            print(f"Predictions: {outputs_.squeeze()}")

            print(f"Hits: {np.sum(outputs_ == labels_).astype(int)}/{len(labels_)}")
            print(f"Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


class MLFlowDLPipeline:
    def __init__(self, model, optimizer, loss_function,
                 train_loader: FactoryLoader, val_loader: FactoryLoader, name=None,
                 challenge="Ch1", patience=5, logits=False):
        self.challenge = challenge
        self.name = name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model = self.model.to(self.device)

        self.optimizer = optimizer
        self.logits = logits
        self.loss_function = loss_function

        self.train_loader = train_loader.get_loader()
        self.val_loader = val_loader.get_loader()

        self.best_model = {
            "model": None,
            "loss": float('inf')
        }

        self.training_losses = []
        self.validation_losses = []
        self.preprocessing_steps = train_loader.get_transformation_steps()
        self.patience = patience
        self.early_stop_counter = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

        mlflow_tracking_dir = "file:///C:/Users/gimes/Src/repos/CADx-Project/mlruns"
        mlflow_tracking_dir = "http://localhost:5000"
        mlflow.set_tracking_uri(mlflow_tracking_dir)
        print(f"MLflow runs will be saved in: {mlflow_tracking_dir}")

        self.save_path = os.path.join(project_root, "2_DL", challenge, "runs")
        self.start_time = datetime.datetime.now()
        self.run_time = None
        self.save_path_id = os.path.join(self.save_path, self.start_time.strftime("%m-%d-%Hh%M") + f"_{name}")

        if not os.path.exists(self.save_path_id):
            os.makedirs(self.save_path_id)

        print(f"Saving results to directory:\n{self.save_path_id}")

    def train(self, epochs, debug=True,
              metrics=['accuracy', 'f1_score', 'recall', "precision", "macro avg"]):
        self.training_losses = []
        self.validation_losses = []

        mlflow.set_experiment(self.challenge)
        with mlflow.start_run(run_name=self.name):
            mlflow.log_params({
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "batch_size": self.train_loader.batch_size,
                "epochs": epochs,
                "loss_function": self.loss_function.__class__.__name__,
                "optimizer": self.optimizer.__class__.__name__,
                "preprocessing_steps": self.preprocessing_steps
            })
            for epoch in range(epochs):
                # Train
                logger.info(f'[EPOCH] {epoch + 1}/{epochs}')
                train_loss = _train(self.model, self.train_loader, self.optimizer, self.loss_function, self.device,
                                    logits=self.logits)
                self.training_losses.append(train_loss)
                logger.info(f'[TRAIN] Epoch {epoch + 1}/{epochs} Loss: {train_loss}')
                mlflow.log_metric("loss_training", train_loss, step=epoch)

                # Validate
                val_loss, report = _validate(self.model, self.val_loader, self.loss_function, self.device,
                                             logits=self.logits)
                self.validation_losses.append(val_loss)
                logger.info(f'[VALIDATE] Epoch {epoch + 1}/{epochs} Loss: {val_loss}')
                mlflow.log_metric("loss_validation", val_loss, step=epoch)

                self.run_time = datetime.datetime.now() - self.start_time
                mlflow.log_metric("run_time", self.run_time.total_seconds(), step=epoch)

                for key, value in report.items():
                    if key not in metrics:
                        continue
                    elif isinstance(value, dict):
                        for metric, score in value.items():
                            if metric not in metrics:
                                continue
                            mlflow.log_metric(f"{metric}_{key}", score, step=epoch)
                    else:
                        mlflow.log_metric(key, value, step=epoch)


                if val_loss < self.best_model['loss']:
                    self.best_model['model'] = copy.deepcopy(self.model)
                    self.best_model['loss'] = val_loss
                    self.save_model()
                    self.early_stop_counter = 0
                    logger.info(f'[SAVE] Best model saved with loss: {val_loss}')
                else:
                    self.early_stop_counter += 1
                    logger.info(f'[EARLY STOP] Counter: {self.early_stop_counter}')
                    if self.early_stop_counter == self.patience:
                        logger.info('[EARLY STOP] Stopping training...')
                        break

                self.plot_losses(save=False, show=debug)

            self.plot_losses(save=True, show=True)
            self.plot_roc_auc(save=True, show=True)

            # Create a sample input for signature inference
            sample_input = next(iter(self.val_loader))[0].to(self.device).cpu().numpy()

            # TODO! fix signature and saving
            wrapped_model = ModelWrapper(self.best_model["model"])
            mlflow.sklearn.log_model(wrapped_model, artifact_path="model",
                                     registered_model_name=self.name,
                                     signature=False  # Disables automatic signature inference
                                     #input_example=sample_input,
                                     )
            mlflow.log_artifact(os.path.join(self.save_path_id, 'losses.png'))
            mlflow.log_artifact(os.path.join(self.save_path_id, 'roc_auc.png'))
            mlflow.log_artifact(os.path.join(self.save_path_id, 'hyperparameters.json'))

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
                predicted = torch.sigmoid(outputs) if self.logits else torch.round(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Generate classification report
        report = classification_report(all_labels, all_predictions, target_names=['nevus', 'other'])
        return report

    def plot_losses(self, save=True, show=True):
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if save:
            plt.savefig(os.path.join(self.save_path_id, 'losses.png'))
        if show:
            plt.show()

    def plot_roc_auc(self, save=True, show=True):
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
            plt.savefig(os.path.join(self.save_path_id, 'roc_auc.png'))
        if show:
            plt.show()

    def save_hyperparameters(self):
        hyperparameters = {
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "batch_size": self.train_loader.batch_size,
            "epochs": len(self.training_losses),
            "name": self.name,
            "preprocessing_steps": self.preprocessing_steps,
            "start_time": self.start_time.strftime("%m-%d-%H:%M"),
            "val_size": len(self.val_loader),
            "train_size": len(self.train_loader)
        }
        with open(os.path.join(self.save_path_id, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparameters, f, indent=4)

    def save_model(self):
        torch.save(self.model, f'{self.save_path_id}/best_model.pt')
        self.save_hyperparameters()