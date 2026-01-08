"""
Generate ROC and Precision-Recall Curves for All 7 Models
Visualizations for test set performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("ROC AND PRECISION-RECALL CURVE GENERATION")
print("="*80)

# Load data
print("\n1. Loading IGD data...")
data = pd.read_excel('data/IGD Database.xlsx')
print(f"   Loaded {len(data)} samples")

# Convert hour columns to numeric using function from ml_prediction_demo.py
def convert_hours_to_numeric(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    mapping = {
        '1 or less': 0.5, '1 to 2': 1.5, '2 to 3': 2.5, '3 to 4': 3.5, '4 to 5': 4.5,
        '5 to 6': 5.5, '6 to 7': 6.5, '7 to 8': 7.5, '8 to 9': 8.5, '9 to 10': 9.5,
        '10 or more': 10.5,
    }
    return mapping.get(val, np.nan)

data['Weekday Hours'] = data['Weekday Hours'].apply(convert_hours_to_numeric)
data['Weekend Hours'] = data['Weekend Hours'].apply(convert_hours_to_numeric)

# Convert other columns to numeric
for col in ['Sleep Quality', 'IGD Total', 'Social', 'Escape']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Define features and target
features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 
           'IGD Total', 'Social', 'Escape']
X = data[features].dropna()

# Convert target to binary
if data['IGD Status'].dtype == 'object':
    y = (data.loc[X.index, 'IGD Status'].astype(str).str.strip() == 'Y').astype(int)
else:
    y = data.loc[X.index, 'IGD Status'].astype(int)

# Remove any NaN in target
mask = ~y.isna()
X = X[mask]
y = y[mask]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"   After removing missing values: {len(X)} samples")
print(f"   IGD Positive: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n2. Training models and collecting predictions...")

# Store results
models_data = []

# 1. Logistic Regression
print("   [1/7] Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
models_data.append(('Logistic Regression', y_prob_lr))

# 2. Random Forest
print("   [2/7] Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, 
                            class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
models_data.append(('Random Forest', y_prob_rf))

# 3. SVM
print("   [3/7] SVM...")
svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
svm.fit(X_train_scaled, y_train)
y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]
models_data.append(('SVM', y_prob_svm))

# 4. Gradient Boosting
print("   [4/7] Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
y_prob_gb = gb.predict_proba(X_test)[:, 1]
models_data.append(('Gradient Boosting', y_prob_gb))

# 5. XGBoost
print("   [5/7] XGBoost...")
xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, 
                    scale_pos_weight=10, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
models_data.append(('XGBoost', y_prob_xgb))

# 6. LightGBM
print("   [6/7] LightGBM...")
lgb = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, 
                     class_weight='balanced', verbose=-1)
lgb.fit(X_train, y_train)
y_prob_lgb = lgb.predict_proba(X_test)[:, 1]
models_data.append(('LightGBM', y_prob_lgb))

# 7. Deep Learning (MLP)
print("   [7/7] Deep Learning (MLP)...")
mlp = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
mlp.fit(X_train_scaled, y_train, epochs=100, batch_size=16, 
        callbacks=[early_stop], verbose=0)
y_prob_mlp = mlp.predict(X_test_scaled, verbose=0).flatten()
models_data.append(('Deep Learning (MLP)', y_prob_mlp))

print("\n3. Generating ROC curves...")

# Create ROC curve plot
plt.figure(figsize=(14, 10))
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#17becf']

for idx, (name, y_prob) in enumerate(models_data):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[idx], lw=3, 
             label=f'{name} (AUC = {roc_auc:.3f})')

# Random classifier baseline
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18, fontweight='bold')
plt.title('ROC Curves - All 7 Models', fontsize=22, fontweight='bold', pad=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='lower right', fontsize=14, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualizations/roc_curves_all_models.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/roc_curves_all_models.png")

print("\n4. Generating Precision-Recall curves...")

# Create Precision-Recall curve plot
plt.figure(figsize=(14, 10))

for idx, (name, y_prob) in enumerate(models_data):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    plt.plot(recall, precision, color=colors[idx], lw=3,
             label=f'{name} (AP = {pr_auc:.3f})')

# Baseline (no-skill)
no_skill = y_test.sum() / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], 'k--', lw=2, 
         label=f'No Skill Baseline (AP = {no_skill:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Sensitivity)', fontsize=18, fontweight='bold')
plt.ylabel('Precision', fontsize=18, fontweight='bold')
plt.title('Precision-Recall Curves - All 7 Models', fontsize=22, fontweight='bold', pad=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best', fontsize=14, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualizations/precision_recall_curves_all_models.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: visualizations/precision_recall_curves_all_models.png")

print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE")
print("="*80)
print(f"\nTest set size: {len(y_test)} samples ({y_test.sum()} positive)")
print("\nFiles saved to visualizations/:")
print("  - roc_curves_all_models.png")
print("  - precision_recall_curves_all_models.png")
print("="*80)
