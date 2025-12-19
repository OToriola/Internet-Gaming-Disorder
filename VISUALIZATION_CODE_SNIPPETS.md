# CODE SNIPPETS FOR DISSERTATION VISUALIZATIONS

This file contains individual Python code snippets that can be run independently to generate each required visualization.

## SETUP (Run this first)

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, average_precision_score, confusion_matrix,
                            roc_curve, precision_recall_curve, auc)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Change to your working directory
os.chdir(r"C:\Users\User\OneDrive - Southampton Solent University\Healthcare")

# Load and preprocess data
df = pd.read_excel("IGD_Project/data/IGD Database.xlsx")

weekday_map = {'1 or less': 1, '2 to 3': 2.5, '4 to 5': 4.5, '6 to 7': 6.5, '8 to 10': 9}
weekend_map = {'1 or less': 1, '2 to 3': 2.5, '4 to 5': 4.5, '6 to 7': 6.5, '8 to 10': 9, '11 or more': 11}

df['Weekday Hours'] = df['Weekday Hours'].map(weekday_map)
df['Weekend Hours'] = df['Weekend Hours'].map(weekend_map)

features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 'IGD Total', 'Social', 'Escape']
target = 'IGD Status'
df = df.dropna(subset=features + [target])
df[target] = (df[target] == 'Y').astype(int)

X = df[features].reset_index(drop=True)
y = df[target].reset_index(drop=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Data loaded: {len(X)} samples")
print(f"[OK] Train: {len(X_train)}, Test: {len(X_test)}")
```

---

## 1. CONFUSION MATRICES VISUALIZATION

```python
# Train all models
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
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
    'LightGBM': LGBMClassifier(class_weight='balanced', verbose=-1, random_state=42)
}

y_preds = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_preds[name] = model.predict(X_test)

# Create confusion matrix visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (name, y_pred) in enumerate(y_preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
               cbar=False, annot_kws={'size': 14})
    
    tn, fp, fn, tp = cm.ravel()
    axes[idx].set_title(f'{name}\nTN={tn}, FP={fp}, FN={fn}, TP={tp}', 
                       fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: confusion_matrices.png")
plt.show()
```

---

## 2. ROC AND PRECISION-RECALL CURVES

```python
# Get probability predictions
y_probs = {}
for name, model in models.items():
    y_probs[name] = model.predict_proba(X_test)[:, 1]

# Create ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curves
for name, y_prob in y_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC={auc_score:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([-0.02, 1.02])
axes[0].set_ylim([-0.02, 1.02])

# Precision-Recall Curves
for name, y_prob in y_probs.items():
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    axes[1].plot(recall_vals, precision_vals, linewidth=2.5, label=f'{name} (PR-AUC={pr_auc:.3f})')

axes[1].set_xlabel('Recall (Sensitivity)', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curves - Model Comparison', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: roc_pr_curves.png")
plt.show()
```

---

## 3. CLINICAL DECISION THRESHOLD ANALYSIS

```python
# Use Random Forest for threshold analysis
y_prob_rf = y_probs['Random Forest']
thresholds = np.arange(0, 1.01, 0.05)

precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:
    y_pred_t = (y_prob_rf >= threshold).astype(int)
    if y_pred_t.sum() == 0:
        precisions.append(0)
        recalls.append(0)
        f1_scores.append(0)
    else:
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred_t, zero_division=0))

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(thresholds, precisions, 'o-', linewidth=2.5, label='Precision', color='#1f77b4', markersize=8)
ax.plot(thresholds, recalls, 's-', linewidth=2.5, label='Recall (Sensitivity)', color='#ff7f0e', markersize=8)
ax.plot(thresholds, f1_scores, '^-', linewidth=2.5, label='F1-Score', color='#2ca02c', markersize=8)

# Highlight recommended threshold
ax.axvline(x=0.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Recommended (0.30)')
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Default (0.50)')

ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Clinical Decision Threshold Analysis\n(Random Forest Model)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('clinical_thresholds.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: clinical_thresholds.png")
plt.show()

# Print recommended thresholds
print("\n" + "="*60)
print("RECOMMENDED CLINICAL THRESHOLDS")
print("="*60)
for t in [0.3, 0.5, 0.7]:
    y_pred_t = (y_prob_rf >= t).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    print(f"Threshold {t}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")
```

---

## 4. FEATURE IMPORTANCE (SHAP VALUES)

```python
import shap

# Train Random Forest model
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Use positive class SHAP values
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Calculate mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': mean_abs_shap
}).sort_values('Importance', ascending=False)

# Plot top 10 features
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))

ax.barh(range(len(feature_importance)), feature_importance['Importance'].values, color=colors)
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['Feature'].values)
ax.set_xlabel('Mean |SHAP Value| (Average Impact on Predictions)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance: Random Forest Model\n(SHAP Explainability)', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: shap_feature_importance.png")
plt.show()

print("\nFeature Importance Ranking:")
print(feature_importance.to_string(index=False))
```

---

## 5. CROSS-VALIDATION ANALYSIS

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_results.append({
        'Model': name,
        'Mean CV Accuracy': scores.mean(),
        'Std Dev': scores.std(),
        'Min': scores.min(),
        'Max': scores.max()
    })
    print(f"{name}:")
    print(f"  Mean: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Range: {scores.min():.4f} to {scores.max():.4f}\n")

cv_df = pd.DataFrame(cv_results)
print("\nCross-Validation Summary:")
print(cv_df.to_string(index=False))

# Save as table
cv_df.to_csv('cross_validation_results.csv', index=False)
print("\n[OK] Saved: cross_validation_results.csv")
```

---

## 6. MODEL PERFORMANCE COMPARISON TABLE

```python
# Generate performance metrics for all models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_test, y_prob),
        'PR-AUC': average_precision_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("MODEL PERFORMANCE - TEST SET")
print("="*80)
print(results_df.to_string(index=False))

# Save to CSV for dissertation
results_df.to_csv('model_performance_test_set.csv', index=False)
print("\n[OK] Saved: model_performance_test_set.csv")
```

---

## QUICK EXECUTION

To run all visualizations at once:

```python
# Run SETUP code above first, then:
exec(open('confusion_matrices').read())  # Run ROC/PR curves code
exec(open('clinical_thresholds').read())  # Run threshold analysis code
exec(open('shap_feature_importance').read())  # Run SHAP code
exec(open('cross_validation_analysis').read())  # Run CV code
exec(open('model_performance').read())  # Run performance table code
```

---

## OUTPUT FILES

All code generates PNG files ready for dissertation:
- `confusion_matrices.png` - Confusion matrices for all 6 models
- `roc_pr_curves.png` - ROC and precision-recall curve comparison
- `clinical_thresholds.png` - Decision threshold analysis
- `shap_feature_importance.png` - Feature importance visualization
- `cross_validation_results.csv` - CV performance metrics
- `model_performance_test_set.csv` - Test set performance metrics

---

## TROUBLESHOOTING

**Issue:** "Module not found"
**Solution:** `pip install shap xgboost lightgbm scikit-learn pandas matplotlib seaborn`

**Issue:** "Data file not found"
**Solution:** Make sure IGD_Project/data/IGD Database.xlsx exists and adjust path as needed

**Issue:** "Memory error"
**Solution:** Reduce n_estimators in RandomForest/XGBoost or use subset of data

---

**Ready to use!** ✅
