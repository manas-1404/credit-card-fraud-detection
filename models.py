import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report, accuracy_score
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from xgboost import XGBClassifier
import optuna

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        """
        Fraud Detection Neural Network

        Args:
            input_size: Number of input features (e.g., 30)
            hidden_sizes: List of hidden layer sizes (e.g., [128, 256])
            dropout_rate: Dropout probability (e.g., 0.3)
        """
        super(NeuralNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Trainer:
    def __init__(self, model, device='cuda', learning_rate=0.001):
        """
        Model trainer

        Args:
            model: Neural network model
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_pr_auc = 0

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'pr_auc': [],
            'recall': [],
            'precision': []
        }

    def train(self, train_loader, val_loader=None, epochs=50, verbose=True):
        """
        Train the model

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of training epochs
            verbose: Print progress
        """
        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader:
                metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(metrics['val_loss'])
                self.history['pr_auc'].append(metrics['pr_auc'])
                self.history['recall'].append(metrics['recall'])
                self.history['precision'].append(metrics['precision'])

                if verbose and (epoch + 1) % 5 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], '
                          f'Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {metrics["val_loss"]:.4f}, '
                          f'PR-AUC: {metrics["pr_auc"]:.4f}, '
                          f'Recall: {metrics["recall"]:.4f}')

                # Save best model
                if metrics['pr_auc'] > self.best_pr_auc:
                    self.best_pr_auc = metrics['pr_auc']
                    self.save('best_model.pth')
                    if verbose and (epoch + 1) % 5 == 0:
                        print(f'New best PR-AUC: {self.best_pr_auc:.4f}')

            elif verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_train_loss:.4f}')

    def predict(self, test_loader):
        """
        Get predictions

        Returns:
            probs: Probability scores
            preds: Binary predictions (threshold=0.5)
            labels: True labels
        """
        self.model.eval()

        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions = (outputs > 0.5).float()

                all_probs.extend(outputs.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        return (
            np.array(all_probs).flatten(),
            np.array(all_preds).flatten(),
            np.array(all_labels).flatten()
        )

    def evaluate(self, test_loader):
        """
        Evaluate model performance

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        val_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                predictions = (outputs > 0.5).float()

                all_probs.extend(outputs.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        probs = np.array(all_probs).flatten()
        preds = np.array(all_preds).flatten()
        labels = np.array(all_labels).flatten()

        metrics = {
            'val_loss': val_loss / len(test_loader),
            'pr_auc': average_precision_score(labels, probs),
            'recall': recall_score(labels, preds, zero_division=0),
            'precision': precision_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0)
        }

        return metrics

    def print_evaluation(self, test_loader):
        """Print detailed evaluation"""
        probs, preds, labels =  self.predict(test_loader)

        pr_auc = average_precision_score(labels, probs)
        recall = recall_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        print('\n' + '=' * 70)
        print('EVALUATION RESULTS')
        print('=' * 70)
        print(f'PR-AUC:    {pr_auc:.4f}')
        print(f'Recall:    {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'F1-Score:  {f1:.4f}')

        print('\nConfusion Matrix:')
        cm = confusion_matrix(labels, preds)
        print(cm)

        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            print(f'\nTrue Negatives:  {tn}')
            print(f'False Positives: {fp}')
            print(f'False Negatives: {fn}')
            print(f'True Positives:  {tp}')

        print('\nClassification Report:')
        print(classification_report(labels, preds, target_names=['Legitimate', 'Fraud']))
        print('=' * 70)

    def plot_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Loss
        ax = axes[0, 0]
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        ax.plot(epochs_range, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            ax.plot(epochs_range, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: PR-AUC
        ax = axes[0, 1]
        if self.history['pr_auc']:
            ax.plot(epochs_range, self.history['pr_auc'], 'g-', linewidth=2)
            ax.axhline(y=self.best_pr_auc, color='r', linestyle='--',
                       label=f'Best: {self.best_pr_auc:.4f}', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('PR-AUC', fontsize=12)
        ax.set_title('Precision-Recall AUC', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Plot 3: Recall
        ax = axes[1, 0]
        if self.history['recall']:
            ax.plot(epochs_range, self.history['recall'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title('Recall (Fraud Catch Rate)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Plot 4: Precision
        ax = axes[1, 1]
        if self.history['precision']:
            ax.plot(epochs_range, self.history['precision'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history saved to {save_path}")
        plt.show()

    def save(self, filepath):
        """Save model weights"""
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        """Load model weights"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f'Model loaded from {filepath}')


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    """

    def __init__(self):
        """
        Initialize base model.
        """
        self.model = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaseModel':
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature dataframe
        """
        pass

    def predict_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature dataframe
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("Model does not support predict_proba")

        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, zero_division: int = 0) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels
            zero_division: Value to return for zero division in metrics
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        y_pred = self.predict(X_test)

        report = classification_report(y_test, y_pred, zero_division=zero_division, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)

        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }

    def print_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series, zero_division: int = 0) -> None:
        """
        Print evaluation metrics.

        Args:
            X_test: Test features
            y_test: Test labels
            zero_division: Value to return for zero division
        """
        results = self.evaluate(X_test, y_test, zero_division)

        print("=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1-Score: {results['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(y_test, results['predictions'], zero_division=zero_division))

    def is_fitted(self) -> bool:
        """
        Check if model has been fitted.
        """
        return self._is_fitted

    def get_model(self):
        """
        Get the underlying sklearn model.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model hyperparameters.

        Args:
            **params: Hyperparameters to set
        """
        pass
