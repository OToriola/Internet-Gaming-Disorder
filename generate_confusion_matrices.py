"""
Generate Confusion Matrix Visualizations for All 7 Models (IGD Dataset)
With LARGE fonts for screenshots and presentations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
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

# Create output directory
OUTPUT_DIR = "visualizations/confusion_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING CONFUSION MATRICES FOR ALL 7 MODELS")
print("=" * 80)

# Load data
print("\n1. Loading IGD data...")
df = pd.read_excel("data/IGD Database.xlsx")

# Convert categorical variables to numeric
weekday_hours_map = {'1 or less': 1, '2 to 3': 2.5, '4 to 5': 4.5, '6 to 7': 6.5, '8 to 10': 9}
weekend_hours_map = {'1 or less': 1, '2 to 3': 2.5, '4 to 5': 4.5, '6 to 7': 6.5, '8 to 10': 9, '11 or more': 11}

df['Weekday Hours'] = df['Weekday Hours'].map(weekday_hours_map)
df['Weekend Hours'] = df['Weekend Hours'].map(weekend_hours_map)

# Select features and target
features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 'IGD Total', 'Social', 'Escape']
target = 'IGD Status'

df = df.dropna(subset=features + [target])
X = df[features]
y = (df[target] == 'Y').astype(int)

print(f"   Loaded {len(X)} samples")
print(f"   IGD Positive: {y.sum()} ({y.mean()*100:.1f}%)")

# Split data
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Calculate class weight
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

print("\n3. Training all 7 models and generating confusion matrices...")
print("   " + "-" * 60)

# Define all 7 models
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, class_weight='balanced', random_state=42))
    ]),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42),
    'LightGBM': LGBMClassifier(class_weight='balanced', n_estimators=100, random_state=42, verbose=-1)
}

confusion_matrices = {}

# Train and predict for traditional ML models
for idx, (name, model) in enumerate(models.items(), 1):
    print(f"   {idx}/7: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

# Add Deep Learning model
print(f"   7/7: Deep Learning (MLP, Tuned)")

# Scale data for DL
scaler_dl = StandardScaler()
X_train_scaled = scaler_dl.fit_transform(X_train)
X_test_scaled = scaler_dl.transform(X_test)

def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units1', min_value=8, max_value=64, step=8), 
                    activation='relu', input_dim=X_train_scaled.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(hp.Int('units2', min_value=4, max_value=32, step=4), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Tuning with reduced epochs for speed
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='igd_kt_dir',
    project_name='igd_tuning',
    overwrite=True
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(X_train_scaled, y_train, epochs=20, validation_split=0.2, 
             callbacks=[early_stop], verbose=0)

best_model = tuner.get_best_models(num_models=1)[0]
y_pred_dl = (best_model.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()
confusion_matrices['Deep Learning (MLP, Tuned)'] = confusion_matrix(y_test, y_pred_dl)

print("   " + "-" * 60)

# Generate individual confusion matrix plots with LARGE fonts
print("\n4. Generating individual confusion matrix visualizations...")
print("   " + "-" * 60)

for idx, (name, cm) in enumerate(confusion_matrices.items(), 1):
    print(f"   {idx}/7: {name}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    
    # Enhance fonts
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=32, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=28, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=28, fontweight='bold')
    
    # Make tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=26)
    
    # Make the text in cells larger
    for text in disp.text_.ravel():
        text.set_fontsize(32)
        text.set_fontweight('bold')
    
    # Adjust colorbar label size
    cbar = disp.im_.colorbar
    cbar.ax.tick_params(labelsize=22)
    
    plt.tight_layout()
    
    # Save with zero-padded numbering
    filename = f"{OUTPUT_DIR}/{idx:02d}_{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("   " + "-" * 60)

# Create combined plot (2x4 grid)
print("\n5. Generating combined confusion matrices plot...")

fig, axes = plt.subplots(2, 4, figsize=(28, 14))
axes = axes.flatten()

for idx, (name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[idx]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N', 'P'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    
    ax.set_title(name, fontsize=24, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted', fontsize=20, fontweight='bold')
    ax.set_ylabel('True', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Make cell text larger
    for text in disp.text_.ravel():
        text.set_fontsize(22)
        text.set_fontweight('bold')

# Hide the last empty subplot
axes[7].axis('off')

fig.suptitle('Confusion Matrices: All 7 Models (IGD Dataset)', 
             fontsize=36, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUTPUT_DIR}/00_all_models_combined.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 00_all_models_combined.png")

# Generate summary statistics
print("\n6. Generating summary statistics...")

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
print("CONFUSION MATRIX SUMMARY STATISTICS")
print("=" * 120)
print(summary_df.to_string(index=False))
print("=" * 120)

# Save summary to CSV
summary_path = f"{OUTPUT_DIR}/confusion_matrix_statistics.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary statistics saved to: {summary_path}")

# Final summary
print("\n" + "=" * 80)
print("✅ CONFUSION MATRIX VISUALIZATIONS COMPLETE!")
print("=" * 80)
print(f"\nGenerated files in '{OUTPUT_DIR}/':")
print(f"  • 00_all_models_combined.png     - All 7 models in one figure")
print(f"  • 01-07_*.png                    - Individual model confusion matrices")
print(f"  • confusion_matrix_statistics.csv - Performance metrics")
print("\nAll Models:")
for idx, name in enumerate(confusion_matrices.keys(), 1):
    print(f"  {idx}. {name}")
print("\nFont sizes optimized for screenshots and presentations!")
print("=" * 80)
