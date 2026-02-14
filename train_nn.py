import pandas as pd
import numpy as np
import os
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import imblearn
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score

from models import NeuralNetwork, Trainer

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

df = pd.read_csv(os.path.join(path, "creditcard.csv"))

df['Amount_Log'] = np.log1p(df['Amount'])
scaler_std = StandardScaler()
df['Amount_Log_Std'] = scaler_std.fit_transform(df[['Amount_Log']])

drop_columns = ["Time", "Amount", "Amount_Log"]
df.drop(columns=drop_columns, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Class'],df['Class'], test_size=0.2, random_state=34666, stratify=df['Class'])

n_legitimate = sum(y_train == 0)
n_fraud = sum(y_train == 1)

print(f"Original frauds: {n_fraud}")

target_frauds_conservative = int(n_fraud * 2)

smote_conservative = SMOTE(sampling_strategy={1: target_frauds_conservative}, random_state=3366)
X_train_1_5x, y_train_1_5x = smote_conservative.fit_resample(X_train, y_train)

print(f"After SMOTE (2x): {sum(y_train_1_5x == 1)} frauds")
print(f"Created {sum(y_train_1_5x == 1) - n_fraud} synthetic frauds")

print("Input feature shape: ", X_train_1_5x.shape)
print("Input label shape: ", y_train_1_5x.shape)

X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_1_5x,y_train_1_5x, test_size=0.2, random_state=34666)

batch_size = 256
epochs = 60
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

nn_model = NeuralNetwork(input_size=29, hidden_sizes=[128, 256], dropout_rate=0.173092)

trainer = Trainer(model=nn_model, device=device, learning_rate=7.936364e-05)

X_train_tensor = torch.FloatTensor(X_train_final.values)
y_train_tensor = torch.FloatTensor(y_train_final).unsqueeze(1)

X_val_tensor = torch.FloatTensor(X_val.values)
y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1)

X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs, verbose=True)

trainer.print_evaluation(test_loader=test_loader)

trainer.plot_history()