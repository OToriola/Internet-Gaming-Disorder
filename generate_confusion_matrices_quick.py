"""
Generate Confusion Matrix Visualizations for All 7 Models + Primary 4 Models (IGD Dataset)
With LARGE fonts for screenshots and presentations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Set style with LARGE fonts
sns.set_style("white")
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 28,
    "axes.labelsize": 26,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 22,
})

OUTPUT_DIR = "visualizations/confusion_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING CONFUSION MATRICES FOR ALL 7 MODELS + PRIMARY 4")
print("=" * 80)

# Load data
print("\n1. Loading IGD data...")
DATA_PATH = "data/IGD Database.xlsx"
df = pd.read_excel(DATA_PATH)

# Convert categorical variables to numeric - MATCH ORIGINAL EXACTLY
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

# Convert other features to numeric
for col in ['Sleep Quality', 'IGD Total', 'Social', 'Escape']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Select features and target
features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 'IGD Total', 'Social', 'Escape']
target = 'IGD Status'

# Drop rows with any missing values in features
X = df[features].dropna()

# Convert target to binary - handle both string and numeric
if df[target].dtype == 'object':
    y = (df.loc[X.index, target].astype(str).str.strip() == 'Y').astype(int)
else:
    y = df.loc[X.index, target].astype(int)

# Remove any NaN in target
mask = ~y.isna()
X = X[mask]
y = y[mask]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"   Loaded {len(X)} samples")
print(f"   IGD Positive: {y.sum()} ({y.mean()*100:.1f}%)")

# Split data
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = 10  # Match original

# Train traditional ML models (6 models) - MATCH ORIGINAL PARAMETERS
print("\n3. Training all 6 traditional ML models...")
print("   " + "-" * 60)

models = {
    'Logistic Regression': ('scaled', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    'Random Forest': ('unscaled', RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)),
    'SVM': ('scaled', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)),
    'Gradient Boosting': ('unscaled', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
    'XGBoost': ('unscaled', XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0)),
    'LightGBM': ('unscaled', LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', verbose=-1, random_state=42))
}

confusion_matrices = {}

for idx, (name, (data_type, model)) in enumerate(models.items(), 1):
    print(f"   {idx}/6: {name}")
    
    if data_type == 'scaled':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

# Train Deep Learning model (7th model) - MATCH ORIGINAL
print(f"   7/7: Deep Learning (MLP)")

mlp_model = Sequential([
    Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

mlp_model.fit(
    X_train_scaled, y_train,
    epochs=100, batch_size=16, validation_split=0.2,
    callbacks=[early_stop], verbose=0
)

y_pred_proba_mlp = mlp_model.predict(X_test_scaled, verbose=0).flatten()
y_pred_dl = (y_pred_proba_mlp > 0.5).astype(int)
confusion_matrices['Deep Learning (MLP)'] = confusion_matrix(y_test, y_pred_dl)

print("   " + "-" * 60)

# Generate individual confusion matrix plots with LARGE fonts
print("\n4. Generating individual confusion matrix visualizations (7 models)...")
print("   " + "-" * 60)

for idx, (name, cm) in enumerate(confusion_matrices.items(), 1):
    print(f"   {idx}/7: {name}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    
    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    ax.set_title(f'{name}\nSens={sensitivity:.2f}, Spec={specificity:.2f}', 
                 fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight('bold')
    
    cbar = disp.im_.colorbar
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    filename = f"{OUTPUT_DIR}/{idx:02d}_{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("   " + "-" * 60)

# Create combined plot (2x4 grid for all 7 models)
print("\n5. Generating combined confusion matrices plot (all 7 models)...")

fig, axes = plt.subplots(2, 4, figsize=(28, 14))
axes = axes.flatten()

for idx, (name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[idx]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N', 'P'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    
    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    ax.set_title(f'{name}\nSens={sensitivity:.2f}, Spec={specificity:.2f}', 
                 fontsize=18, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted', fontsize=16, fontweight='bold')
    ax.set_ylabel('True', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight('bold')

# Hide the last empty subplot
axes[7].axis('off')

fig.suptitle('Confusion Matrices: All 7 Models (IGD Dataset)', 
             fontsize=22, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUTPUT_DIR}/00_all_7_models_combined.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 00_all_7_models_combined.png")

# Create PRIMARY MODELS plot (2x2 grid: LightGBM, Random Forest, XGBoost, Gradient Boosting)
print("\n6. Generating PRIMARY MODELS confusion matrices (4 best models)...")

primary_model_names = ['LightGBM', 'Random Forest', 'XGBoost', 'Gradient Boosting']
primary_cms = {name: cm for name, cm in confusion_matrices.items() if name in primary_model_names}

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (name, cm) in enumerate(primary_cms.items()):
    ax = axes[idx]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    
    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    ax.set_title(f'{name}\nSens={sensitivity:.2f}, Spec={specificity:.2f}', 
                 fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_fontweight('bold')

fig.suptitle('Confusion Matrices: Primary Models (4 Best Performing)', 
             fontsize=22, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUTPUT_DIR}/00_primary_4_models.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 00_primary_4_models.png")

# Generate summary statistics
print("\n7. Generating summary statistics...")

summary_stats = []
for name, cm in confusion_matrices.items():
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    summary_stats.append({
        'Model': name,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity
    })

summary_df = pd.DataFrame(summary_stats)

print("\n" + "=" * 120)
print("CONFUSION MATRIX SUMMARY STATISTICS (7 Models)")
print("=" * 120)
print(summary_df.to_string(index=False))
print("=" * 120)

summary_path = f"{OUTPUT_DIR}/confusion_matrix_statistics_all_7.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary statistics saved to: {summary_path}")

# Save primary models summary
primary_summary = summary_df[summary_df['Model'].isin(primary_model_names)]
primary_path = f"{OUTPUT_DIR}/confusion_matrix_statistics_primary_4.csv"
primary_summary.to_csv(primary_path, index=False)
print(f"✓ Primary models summary saved to: {primary_path}")

# Final summary
print("\n" + "=" * 80)
print("✅ CONFUSION MATRIX VISUALIZATIONS COMPLETE!")
print("=" * 80)
print(f"\nGenerated files in '{OUTPUT_DIR}/':")
print(f"\n  📊 MAIN FIGURES (for dissertation):")
print(f"  • 00_primary_4_models.png        - PRIMARY: 4 best models (2x2 grid)")
print(f"  • 00_all_7_models_combined.png   - All 7 models (2x4 grid)")
print(f"\n  📁 Individual model matrices (7 files):")
print(f"  • 01-07_*.png                    - One per model")
print(f"\n  📈 Summary statistics:")
print(f"  • confusion_matrix_statistics_all_7.csv    - All 7 models metrics")
print(f"  • confusion_matrix_statistics_primary_4.csv - Primary 4 models metrics")
print("\n" + "=" * 80)
print("All 7 Models:")
for idx, name in enumerate(confusion_matrices.keys(), 1):
    is_primary = " ⭐" if name in primary_model_names else ""
    print(f"  {idx}. {name}{is_primary}")
print("\n⭐ = Primary Model (Best 4)")
print("=" * 80)
