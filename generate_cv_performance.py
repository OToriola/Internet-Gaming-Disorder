"""
Generate Cross-Validation Performance Table (5-fold stratified)
For all 7 models to assess generalization and stability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
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
print("STRATIFIED 5-FOLD CROSS-VALIDATION - ALL 7 MODELS")
print("="*80)

# Load data
print("\n1. Loading IGD data...")
df = pd.read_excel("data/IGD Database.xlsx")

# Convert hours
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

df['Weekday Hours'] = df['Weekday Hours'].apply(convert_hours_to_numeric)
df['Weekend Hours'] = df['Weekend Hours'].apply(convert_hours_to_numeric)

for col in ['Sleep Quality', 'IGD Total', 'Social', 'Escape']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 'IGD Total', 'Social', 'Escape']
target = 'IGD Status'

X = df[features].dropna()

if df[target].dtype == 'object':
    y = (df.loc[X.index, target].astype(str).str.strip() == 'Y').astype(int)
else:
    y = df.loc[X.index, target].astype(int)

mask = ~y.isna()
X = X[mask]
y = y[mask]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"   Total samples: {len(X)}")
print(f"   IGD Positive: {y.sum()} ({y.mean()*100:.1f}%)")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Setup 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n2. Running 5-fold cross-validation for each model...")
print("   This may take a few minutes...")

results = []

# Logistic Regression
print("\n   [1/7] Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
results.append({
    'Model': 'Logistic Regression',
    'Mean Accuracy': scores.mean(),
    'Standard Deviation': scores.std(),
    'Fold Scores': scores
})

# Random Forest
print("   [2/7] Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
results.append({
    'Model': 'Random Forest',
    'Mean Accuracy': scores.mean(),
    'Standard Deviation': scores.std(),
    'Fold Scores': scores
})

# SVM
print("   [3/7] SVM...")
svm = SVC(class_weight='balanced', random_state=42)
scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
results.append({
    'Model': 'SVM',
    'Mean Accuracy': scores.mean(),
    'Standard Deviation': scores.std(),
    'Fold Scores': scores
})

# Gradient Boosting
print("   [4/7] Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
scores = cross_val_score(gb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
results.append({
    'Model': 'Gradient Boosting',
    'Mean Accuracy': scores.mean(),
    'Standard Deviation': scores.std(),
    'Fold Scores': scores
})

# XGBoost
print("   [5/7] XGBoost...")
xgb = XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=10, random_state=42, verbosity=0)
scores = cross_val_score(xgb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
results.append({
    'Model': 'XGBoost',
    'Mean Accuracy': scores.mean(),
    'Standard Deviation': scores.std(),
    'Fold Scores': scores
})

# LightGBM
print("   [6/7] LightGBM...")
lgb = LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, verbose=-1)
scores = cross_val_score(lgb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
results.append({
    'Model': 'LightGBM',
    'Mean Accuracy': scores.mean(),
    'Standard Deviation': scores.std(),
    'Fold Scores': scores
})

# Deep Learning (MLP) - manual CV due to Keras
print("   [7/7] Deep Learning (MLP)...")
mlp_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = Sequential([
        Dense(32, activation='relu', input_dim=X_scaled.shape[1]),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=16, 
              callbacks=[early_stop], verbose=0)
    
    _, acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    mlp_scores.append(acc)
    print(f"      Fold {fold+1}/5: {acc:.4f}")

mlp_scores = np.array(mlp_scores)
results.append({
    'Model': 'Deep Learning (MLP)',
    'Mean Accuracy': mlp_scores.mean(),
    'Standard Deviation': mlp_scores.std(),
    'Fold Scores': mlp_scores
})

# Create results table
print("\n" + "="*80)
print("TABLE 10: GENERALISATION PERFORMANCE AND MODEL STABILITY")
print("="*80)

df_results = pd.DataFrame(results)
df_results['Mean Accuracy %'] = (df_results['Mean Accuracy'] * 100).round(2)
df_results['Std Dev %'] = (df_results['Standard Deviation'] * 100).round(2)

print(f"\n{'Model':<25} {'Mean Accuracy':<15} {'Std Deviation':<15}")
print("-" * 55)
for _, row in df_results.iterrows():
    print(f"{row['Model']:<25} {row['Mean Accuracy %']:>6.2f}%          {row['Std Dev %']:>6.2f}%")

print("\n" + "="*80)

# Detailed fold scores
print("\n\nDETAILED FOLD SCORES:")
print("="*80)
for _, row in df_results.iterrows():
    print(f"\n{row['Model']}:")
    for i, score in enumerate(row['Fold Scores'], 1):
        print(f"   Fold {i}: {score:.4f}")

# Save to CSV
output_df = df_results[['Model', 'Mean Accuracy', 'Standard Deviation']].copy()
output_df['Mean Accuracy'] = output_df['Mean Accuracy'].round(4)
output_df['Standard Deviation'] = output_df['Standard Deviation'].round(4)
output_df.to_csv('visualizations/cv_performance_table10.csv', index=False)

print("\n" + "="*80)
print("✓ Results saved to: visualizations/cv_performance_table10.csv")
print("="*80)
