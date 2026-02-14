import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report


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
        self.results = {}

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

            # Validation
            if val_loader and verbose and (epoch + 1) % 5 == 0:
                metrics = self.evaluate(val_loader)
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'PR-AUC: {metrics["pr_auc"]:.4f}, '
                      f'Recall: {metrics["recall"]:.4f}')

                # Save best model
                if metrics['pr_auc'] > self.best_pr_auc:
                    self.best_pr_auc = metrics['pr_auc']
                    self.save('best_model.pth')
                    print(f'  ✓ New best PR-AUC: {self.best_pr_auc:.4f}')

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

        self.results = (
            np.array(all_probs).flatten(),
            np.array(all_preds).flatten(),
            np.array(all_labels).flatten()
        )

        return self.results

    def evaluate(self, test_loader):
        """
        Evaluate model performance

        Returns:
            Dictionary of metrics
        """
        probs, preds, labels =  self.predict(test_loader)

        metrics = {
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

        print('\nEvaluation Results:')
        print(f'PR-AUC:    {pr_auc:.4f}')
        print(f'Recall:    {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'F1-Score:  {f1:.4f}')

        print('\nConfusion Matrix:')
        cm = confusion_matrix(labels, preds)
        print(cm)

        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            print(f'True Negatives:  {tn}')
            print(f'False Positives: {fp}')
            print(f'False Negatives: {fn}')
            print(f'True Positives:  {tp}')

        print('\nClassification Report:')
        print(classification_report(labels, preds, target_names=['Legitimate', 'Fraud']))

    def save(self, filepath):
        """Save model weights"""
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        """Load model weights"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f'Model loaded from {filepath}')