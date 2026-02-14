# Credit Card Fraud Detection

A deep learning approach to detecting fraudulent credit card transactions using PyTorch. This project tackles the challenge of **extreme class imbalance** in fraud detection, where fraudulent transactions represent only 0.172% of all transactions. The final model achieves **90.59% precision** and **78.57% recall**, correctly identifying fraudulent transactions while maintaining an extremely low false positive rate of just 0.014%.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Challenges](#challenges)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup](#setup)

## Problem Statement

Credit card fraud detection is a critical real-world problem where:
- **False negatives** (missing fraud) result in financial losses
- **False positives** (flagging legitimate transactions) frustrate customers and create operational overhead
- The data is **extremely imbalanced** - fraud is rare by nature

This project builds a neural network classifier that balances these competing objectives while handling severe class imbalance.

## Dataset

[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) - European cardholders' transactions from September 2013.

### Dataset Statistics
- **284,807** total transactions
- **492** fraudulent transactions (**0.172%** of all transactions)
- **30 features**: 28 PCA-transformed features (V1-V28), Time, Amount
- **Highly imbalanced** binary classification problem

### Feature Description
- **V1-V28**: PCA-transformed features (to protect user privacy, original features are not provided)
- **Time**: Seconds elapsed between this transaction and the first transaction in the dataset
- **Amount**: Transaction amount (this feature is highly skewed)
- **Class**: Target variable (1 = fraud, 0 = legitimate)

## Challenges

### 1. Extreme Class Imbalance

With only 0.172% fraud, a naive model could achieve 99.83% accuracy by predicting "legitimate" for every transaction while being completely useless at detecting fraud. A baseline model that always predicts legitimate would have 99.83% accuracy but 0% fraud detection rate, which is why we focus on Precision-Recall AUC (PR-AUC) and recall instead of accuracy.

### 2. Feature Skewness
The `Amount` feature has extreme skewness - most transactions are small, but some are very large. This can cause:
- Gradient instability during training
- The model overfitting to large transactions
- Poor generalization

### 3. Cost-Sensitive Errors
Not all errors are equal:
- **False Negative** (missing fraud): Direct financial loss
- **False Positive** (flagging legitimate): Customer frustration, operational cost
- Need to balance recall (catch fraud) vs precision (avoid false alarms)

## Approach

### 1. Data Preprocessing

#### Feature Engineering

The Amount feature was log-transformed using np.log1p (log1p handles zero values) to reduce skewness, then standard scaled using StandardScaler. The original Amount distribution was highly right-skewed with few very large values, but after log transformation it became more normally distributed. This helps the neural network learn better patterns without being dominated by outliers. Features dropped from the final dataset include Time, Amount, and Amount_Log, keeping only the scaled log-transformed version.

### 2. Handling Class Imbalance: SMOTE Experimentation

SMOTE (Synthetic Minority Oversampling Technique) creates synthetic fraud samples by interpolating between existing fraud cases. I experimented with different SMOTE ratios and found that 2x SMOTE performed best with the highest precision (90.6%) and fewest false positives (8). The 2.5x SMOTE ratio caught 2 more frauds but showed signs of overfitting, with a validation PR-AUC of 0.9687 but test PR-AUC of only 0.8220 (gap of 0.15). Conservative oversampling is better because too many synthetic samples cause the model to overfit to synthetic patterns rather than real fraud.

| SMOTE Ratio | Training Frauds | Test PR-AUC | Precision | Recall | False Positives |
|-------------|----------------|-------------|-----------|--------|-----------------|
| 1.5x | 591 | 0.8426 | 88.5% | 78.6% | 10 |
| **2.0x** | **788** | **0.8303** | **90.6%** | **78.6%** | **8** |
| 2.5x | 985 | 0.8220 | 88.8% | 80.6% | 10 |

### 3. Train/Validation/Test Split Strategy

```
Original Data (284,807 transactions)
    |
Train/Test Split (80/20, stratified)
    |
Train (227,845) ---> SMOTE ---> Train with synthetic frauds (228,239)
    |                                   |
    |                           Train/Val Split (80/20, stratified)
    |                                   |
    |                           Train (182,591) | Validation (45,648)
    |
Test (56,962) <--- Never seen during training or validation
```

**Critical**: SMOTE is applied **only to training data** to prevent data leakage.

### 4. Evaluation Metrics

For imbalanced classification, I prioritized PR-AUC (Precision-Recall AUC) as the best metric for imbalanced data since it focuses on the minority class (fraud) and is not affected by the large number of legitimate transactions. Recall measures what percentage of actual frauds are caught, which is critical for minimizing financial losses. Precision measures how often we are correct when we flag fraud, which is important for minimizing false alarms. Accuracy is not used as a primary metric because a model predicting "all legitimate" would get 99.83% accuracy but 0% fraud detection, making accuracy misleading when classes are imbalanced.

## Model Architecture

### Neural Network Design

```
Input Layer (29 features)
    |
Hidden Layer 1: Linear(29 -> 128) -> ReLU -> Dropout(0.173)
    |
Hidden Layer 2: Linear(128 -> 256) -> ReLU -> Dropout(0.173)
    |
Output Layer: Linear(256 -> 1) -> Sigmoid
```

### Hyperparameter Optimization

Used **Optuna** for automated hyperparameter tuning across 50 trials:

| Hyperparameter | Search Space | Optimal Value |
|----------------|--------------|---------------|
| Learning Rate | [1e-5, 1e-2] | 7.94e-05 |
| Dropout Rate | [0.1, 0.5] | 0.173 |
| Hidden Layer 1 | [64, 256] | 128 |
| Hidden Layer 2 | [64, 512] | 256 |
| Batch Size | [32, 512] | 256 |

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy (BCELoss)
- **Epochs**: 60
- **Early Stopping**: Save best model based on validation PR-AUC
- **Device**: CUDA (GPU acceleration)

## Results

### Training History

![Training History](all_models/nn_training_history.png)

*The plot shows smooth convergence without severe overfitting - train and validation losses track closely.*

### Training Progression

| Epoch | Train Loss | Val Loss | Val PR-AUC | Val Recall |
|-------|-----------|----------|------------|------------|
| 5 | 0.0054 | 0.0049 | 0.8373 | 80.9% |
| 15 | 0.0037 | 0.0038 | 0.8930 | 82.7% |
| 30 | 0.0025 | 0.0034 | 0.9166 | 85.8% |
| 45 | 0.0019 | 0.0034 | 0.9237 | 85.2% |
| 60 | 0.0016 | 0.0033 | **0.9270** | 85.8% |

### Final Test Set Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **PR-AUC** | **0.8303** | Strong performance on imbalanced data |
| **Precision** | **90.59%** | When flagged as fraud, correct 90.6% of the time |
| **Recall** | **78.57%** | Catches 78.6% of all fraud cases |
| **F1-Score** | **0.8415** | Balanced harmonic mean of precision/recall |
| **Accuracy** | 99.95% | (Not meaningful for imbalanced data) |

### Confusion Matrix

|  | Predicted Legitimate | Predicted Fraud |
|---|---------------------|-----------------|
| **Actual Legitimate (56,864)** | 56,856 | 8 |
| **Actual Fraud (98)** | 21 | 77 |

### Real-World Impact

The model caught 77 frauds out of 98 total frauds in the test set, while only flagging 8 legitimate transactions as false alarms out of 56,864 legitimate transactions. This translates to a false positive rate of just 0.014%, meaning customers would barely notice any incorrect fraud alerts. The 21 missed frauds would require manual review or additional features to improve detection further.

## Project Structure

```
Credit-Card-Fraud-Detection/
├── models.py                        # NeuralNetwork and Trainer classes
├── train_nn.py                      # Main training script
├── NN_hyperparameter_tuning.ipynb   # Optuna optimization notebook
├── all_models/
│   ├── best_model.pth              # Best model weights
│   ├── nn_training_history.png     # Training curves
│   └── nn_optuna_study.pkl         # Hyperparameter search results
├── .gitignore
└── README.md
```

## Setup

### Installation

```bash
# Clone repository
git clone https://github.com/manas-1404/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install torch numpy pandas scikit-learn imbalanced-learn kagglehub matplotlib seaborn optuna
```

### Training

```bash
python train_nn.py
```

The script will:
1. Download the dataset from Kaggle (requires kagglehub authentication)
2. Preprocess features (log transform, scaling)
3. Apply SMOTE to training data
4. Train the neural network
5. Evaluate on test set and display results

### Hyperparameter Tuning

```bash
jupyter notebook NN_hyperparameter_tuning.ipynb
```

## Tech Stack

- **PyTorch** - Deep learning framework
- **scikit-learn** - Preprocessing, splitting, metrics
- **imbalanced-learn** - SMOTE oversampling
- **Optuna** - Hyperparameter optimization
- **Kaggle Hub** - Dataset management
- **Matplotlib/Seaborn** - Visualization

## Author

Manas - [GitHub](https://github.com/manas-1404)

---
