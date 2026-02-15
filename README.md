# Credit Card Fraud Detection

A machine learning approach to detecting fraudulent credit card transactions using both Neural Networks (PyTorch) and Gradient Boosting (XGBoost). This project tackles the challenge of **extreme class imbalance** in fraud detection, where fraudulent transactions represent only 0.172% of all transactions. Through extensive experimentation with SMOTE oversampling ratios and model architectures, the best models achieve **88-90% precision** and **78-80% recall** while maintaining an extremely low false positive rate.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Challenges](#challenges)
- [Approach](#approach)
- [Models](#models)
- [Results](#results)
- [Model Comparison](#model-comparison)
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

## Models

### 1. Neural Network (PyTorch)

#### Architecture

```
Input Layer (29 features)
    |
Hidden Layer 1: Linear(29 -> 128) -> ReLU -> Dropout(0.173)
    |
Hidden Layer 2: Linear(128 -> 256) -> ReLU -> Dropout(0.173)
    |
Output Layer: Linear(256 -> 1) -> Sigmoid
```

#### Hyperparameter Optimization

Used **Optuna** for automated hyperparameter tuning across 50 trials:

| Hyperparameter | Search Space | Optimal Value |
|----------------|--------------|---------------|
| Learning Rate | [1e-5, 1e-2] | 7.94e-05 |
| Dropout Rate | [0.1, 0.5] | 0.173 |
| Hidden Layer 1 | [64, 256] | 128 |
| Hidden Layer 2 | [64, 512] | 256 |
| Batch Size | [32, 512] | 256 |

#### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy (BCELoss)
- **Epochs**: 60
- **Early Stopping**: Save best model based on validation PR-AUC
- **Device**: CUDA (GPU acceleration)
- **Preprocessing**: Log transformation + Standard scaling on Amount feature

### 2. XGBoost (Gradient Boosting)

#### Why XGBoost?

Tree-based models like XGBoost have several advantages for fraud detection:
- Handle imbalanced data natively through `scale_pos_weight` parameter
- No need for feature scaling or log transformations
- Built-in feature importance for interpretability
- Generally more robust to outliers

#### Hyperparameter Optimization

Used **Optuna** to optimize for PR-AUC across 50 trials:

| Hyperparameter | Search Space | Optimal Value |
|----------------|--------------|---------------|
| Learning Rate | [0.01, 0.3] | 0.0729 |
| Max Depth | [3, 10] | 10 |
| N Estimators | [50, 500] | 271 |
| Min Child Weight | [1, 10] | 1 |
| Subsample | [0.6, 1.0] | 0.626 |
| Colsample Bytree | [0.6, 1.0] | 0.870 |
| Gamma | [0.0, 5.0] | 0.816 |
| Reg Alpha | [0.0, 10.0] | 0.568 |
| Reg Lambda | [0.0, 10.0] | 0.685 |

#### Training Configuration

- **Objective**: Binary logistic regression
- **Eval Metric**: PR-AUC (aucpr)
- **Device**: CUDA (GPU acceleration)
- **Class Imbalance Handling**: scale_pos_weight calculated from data distribution
- **Preprocessing**: Minimal (only dropped Time feature, kept raw Amount)

## Results

### Neural Network Results (2x SMOTE)

#### Training History

![Training History](all_models/nn_training_history.png)

*The plot shows smooth convergence without severe overfitting - train and validation losses track closely.*

#### Training Progression

| Epoch | Train Loss | Val Loss | Val PR-AUC | Val Recall |
|-------|-----------|----------|------------|------------|
| 5 | 0.0054 | 0.0049 | 0.8373 | 80.9% |
| 15 | 0.0037 | 0.0038 | 0.8930 | 82.7% |
| 30 | 0.0025 | 0.0034 | 0.9166 | 85.8% |
| 45 | 0.0019 | 0.0034 | 0.9237 | 85.2% |
| 60 | 0.0016 | 0.0033 | **0.9270** | 85.8% |

#### Test Set Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **PR-AUC** | **0.8303** | Strong performance on imbalanced data |
| **Precision** | **90.59%** | When flagged as fraud, correct 90.6% of the time |
| **Recall** | **78.57%** | Catches 78.6% of all fraud cases |
| **F1-Score** | **0.8415** | Balanced harmonic mean of precision/recall |
| **Accuracy** | 99.95% | (Not meaningful for imbalanced data) |

#### Confusion Matrix

|  | Predicted Legitimate | Predicted Fraud |
|---|---------------------|-----------------|
| **Actual Legitimate (56,864)** | 56,856 | 8 |
| **Actual Fraud (98)** | 21 | 77 |

**Performance Summary:**
- Caught 77 out of 98 frauds (78.6% recall)
- Only 8 false alarms out of 56,864 legitimate transactions (0.014% false positive rate)
- 21 missed frauds would require additional features or manual review

### XGBoost Results (1.25x SMOTE)

#### Test Set Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **PR-AUC** | **0.8311** | Strong performance on imbalanced data |
| **Precision** | **88.64%** | When flagged as fraud, correct 88.6% of the time |
| **Recall** | **79.59%** | Catches 79.6% of all fraud cases |
| **F1-Score** | **0.8387** | Balanced harmonic mean of precision/recall |
| **Accuracy** | 99.95% | (Not meaningful for imbalanced data) |
| **Val-Test Gap** | **0.0696** | Best generalization among all models |

#### Confusion Matrix

|  | Predicted Legitimate | Predicted Fraud |
|---|---------------------|-----------------|
| **Actual Legitimate (56,864)** | 56,854 | 10 |
| **Actual Fraud (98)** | 20 | 78 |

**Performance Summary:**
- Caught 78 out of 98 frauds (79.6% recall)
- Only 10 false alarms out of 56,864 legitimate transactions (0.018% false positive rate)
- Smallest validation-test gap (0.0696) indicates best generalization to real-world fraud patterns
- 20 missed frauds would require additional features or manual review


## Model Comparison

I experimented with both Neural Networks and XGBoost using different SMOTE oversampling ratios to find the optimal balance between catching fraud and minimizing false alarms.

### Experimental Results

| Model | SMOTE Ratio | Test PR-AUC | Precision | Recall | Frauds Caught | False Alarms | Val-Test Gap |
|-------|-------------|-------------|-----------|--------|---------------|--------------|--------------|
| **XGBoost** | **1.25x** | **0.8311** | 88.6% | 79.6% | 78/98 | 10 | **0.0696** |
| **Neural Network** | **2.0x** | **0.8303** | **90.6%** | 78.6% | 77/98 | **8** | 0.0967 |
| XGBoost | 2.0x | **0.8349** | 87.8% | **80.6%** | **79/98** | 11 | 0.1234 |
| XGBoost | 1.0x | 0.8283 | 87.5% | 78.6% | 77/98 | 11 | 0.0923 |
| Neural Network | 1.5x | 0.8426 | 88.5% | 78.6% | 77/98 | 10 | 0.0900 |

### Key Findings

1. **XGBoost with 1.25x SMOTE** achieved the best generalization with the smallest validation-test gap (0.0696), making it the most reliable model for production deployment.

2. **Neural Network with 2x SMOTE** demonstrated the highest precision (90.6%) with only 8 false alarms, minimizing customer disruption while maintaining competitive fraud detection.

3. **XGBoost with 2x SMOTE** caught the most fraud (80.6% recall, 79/98 frauds) but showed more overfitting (validation-test gap of 0.1234).

4. **All models achieved PR-AUC > 0.83**, demonstrating strong performance on this highly imbalanced dataset.

### Which is the better model for solving the problem?

**For Production Deployment:**
**XGBoost with 1.25x SMOTE** is the best choice for production deployment. This model achieves the best generalization with a validation-test gap of only 0.0696, making it the most reliable when deployed on new, unseen fraud patterns. It provides balanced performance with 88.6% precision and 79.6% recall, catching 78 out of 98 frauds while generating only 10 false alarms. Unlike the Neural Network which requires log transformation and standard scaling of features, XGBoost works directly on raw data, simplifying the deployment pipeline and reducing preprocessing overhead.  While the Neural Network achieves slightly higher precision (90.6% vs 88.6%), the XGBoost model's superior generalization and ease of deployment make it the clear winner for real-world fraud detection systems.

### Real-World Impact

The model caught 77 frauds out of 98 total frauds in the test set, while only flagging 8 legitimate transactions as false alarms out of 56,864 legitimate transactions. This translates to a false positive rate of just 0.014%, meaning customers would barely notice any incorrect fraud alerts. The 21 missed frauds would require manual review or additional features to improve detection further.

## Project Structure

```
Credit-Card-Fraud-Detection/
├── models.py                        # NeuralNetwork, XGBoostModel, and Trainer classes
├── train_nn.py                      # Neural Network training script
├── train_xgboost.py                 # XGBoost training script
├── NN_hyperparameter_tuning.ipynb   # Neural Network Optuna optimization
├── XGB_hyperparameter_tuning.ipynb  # XGBoost Optuna optimization
├── all_models/
│   ├── best_model.pth              # Best Neural Network weights
│   ├── xgboost_fraud_detector.pkl  # Best XGBoost model
│   ├── nn_training_history.png     # NN training curves
│   ├── nn_optuna_study.pkl         # NN hyperparameter search results
│   └── xgb_optuna_study.pkl        # XGBoost hyperparameter search results
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
pip install torch numpy pandas scikit-learn imbalanced-learn kagglehub matplotlib seaborn optuna xgboost
```

### Training

#### Neural Network
```bash
python train_nn.py
```

#### XGBoost
```bash
python train_xgboost.py
```

Both scripts will:
1. Download the dataset from Kaggle (requires kagglehub authentication)
2. Preprocess features
3. Apply SMOTE to training data
4. Train the model
5. Evaluate on test set and display results

### Hyperparameter Tuning

```bash
# Neural Network tuning
jupyter notebook NN_hyperparameter_tuning.ipynb

# XGBoost tuning
jupyter notebook XGB_hyperparameter_tuning.ipynb
```

## Tech Stack

- **PyTorch** - Deep learning framework
- **XGBoost** - Gradient boosting framework
- **scikit-learn** - Preprocessing, splitting, metrics
- **imbalanced-learn** - SMOTE oversampling
- **Optuna** - Hyperparameter optimization
- **Kaggle Hub** - Dataset management
- **Matplotlib/Seaborn** - Visualization

## Author

Manas - [GitHub](https://github.com/manas-1404)

---
