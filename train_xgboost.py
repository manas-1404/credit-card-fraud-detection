import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

from models import XGBoostModel

BEST_PARAMS = {
    'n_estimators': 271,
    'learning_rate': 0.07289590286774396,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.6259989306417418,
    'colsample_bytree': 0.870047895607214,
    'gamma': 0.816411782606675,
    'reg_alpha': 0.567984473554783,
    'reg_lambda': 0.6847738823413013,
    'random_state': 355533355533,
    'use_gpu': True,
    'n_jobs': -1,
    'verbosity': 1
}

RANDOM_STATE = 34666
TEST_SIZE = 0.2

print("Downloading dataset...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(os.path.join(path, "creditcard.csv"))

print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {sum(df['Class'] == 1)} ({sum(df['Class'] == 1) / len(df) * 100:.3f}%)")

# Drop Time column (not useful for fraud detection)
df.drop(columns=["Time"], inplace=True)

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

print(f"\nFeatures: {list(X.columns)}")
print(f"Number of features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train set (before SMOTE): {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train frauds (before SMOTE): {sum(y_train == 1)}")
print(f"Test frauds: {sum(y_test == 1)}")

print("\n" + "="*70)
print("APPLYING SMOTE (1.25x) TO TRAINING DATA")
print("="*70)

n_fraud_original = sum(y_train == 1)
target_frauds = int(n_fraud_original * 1.25)

print(f"Original frauds: {n_fraud_original}")
print(f"Target frauds (1.25x): {target_frauds}")

smote = SMOTE(sampling_strategy={1: target_frauds}, random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: {sum(y_train_smote == 1)} frauds")
print(f"Synthetic frauds created: {sum(y_train_smote == 1) - n_fraud_original}")
print(f"New training set size: {X_train_smote.shape[0]} samples")

print("\nSplitting SMOTE training data into train and validation...")

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_smote, y_train_smote,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train_smote
)

print(f"Final train set: {X_train_final.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Train frauds: {sum(y_train_final == 1)}")
print(f"Val frauds: {sum(y_val == 1)}")

n_legitimate = sum(y_train_final == 0)
n_fraud = sum(y_train_final == 1)
scale_pos_weight = n_legitimate / n_fraud

print(f"\nClass imbalance (after SMOTE, final training set):")
print(f"  Legitimate: {n_legitimate:,}")
print(f"  Fraud: {n_fraud:,}")
print(f"  Ratio: {scale_pos_weight:.2f}:1")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

print("\n" + "="*70)
print("TRAINING XGBOOST MODEL")
print("="*70)

model_params = BEST_PARAMS.copy()
model_params['scale_pos_weight'] = scale_pos_weight

xgb_model = XGBoostModel(**model_params)

print("\nModel configuration:")
for key, value in model_params.items():
    if key not in ['random_state', 'n_jobs', 'verbosity', 'use_gpu']:
        print(f"  {key}: {value}")

xgb_model.fit(X_train_final, y_train_final)

print("\n" + "="*70)
print("TEST SET EVALUATION")
print("="*70)

xgb_model.print_evaluation(X_test, y_test)

model_path = "xgboost_fraud_detector.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)

print(f"Model saved to: {model_path}")

val_results = xgb_model.evaluate(X_val, y_val)
test_results = xgb_model.evaluate(X_test, y_test)

print(f"\nValidation Metrics:")
print(f"  PR-AUC:    {val_results['pr_auc']:.4f}")
print(f"  Precision: {val_results['precision']:.4f}")
print(f"  Recall:    {val_results['recall']:.4f}")
print(f"  F1-Score:  {val_results['f1']:.4f}")

print(f"\nTest Metrics:")
print(f"  PR-AUC:    {test_results['pr_auc']:.4f}")
print(f"  Precision: {test_results['precision']:.4f}")
print(f"  Recall:    {test_results['recall']:.4f}")
print(f"  F1-Score:  {test_results['f1']:.4f}")

print(f"\nValidation-Test Gap:")
print(f"  PR-AUC gap: {abs(val_results['pr_auc'] - test_results['pr_auc']):.4f}")

cm = test_results['confusion_matrix']
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTest Set Confusion Matrix:")
    print(f"  Frauds caught: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"  False alarms: {fp}/{fp+tn} ({fp/(fp+tn)*100:.3f}%)")
