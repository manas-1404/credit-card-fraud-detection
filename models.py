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

