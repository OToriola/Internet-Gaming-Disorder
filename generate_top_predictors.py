"""
Top Predictors of Mental Health Risk using Machine Learning (NSCH 2023)
Builds logistic regression model and visualizes feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = "data/nsch_2023e_topical.csv"
OUTPUT_DIR = "visualizations"

print("="*80)
print("TOP PREDICTORS OF MENTAL HEALTH RISK (NSCH 2023)")
print("="*80)

# Load data
print("\n1. Loading NSCH data...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded: {len(df):,} children")

# Extract features and target
print("\n2. Extracting features and target variable...")

# Features
feature_dict = {
    'Screen Time': 'SCREENTIME',
    'Sleep Hours': 'HOURSLEEP',
    'Physical Activity': 'PHYSACTIV',
    'Age': 'SC_AGE_YEARS',
    'Sex': 'SC_SEX',  # 1=Male, 2=Female
}

# Target: Any mental health condition
target_cols = {
    'Depression/Anxiety': 'K2Q32A',
    'Behavioral Problems': 'K2Q33A',
    'ADHD': 'K2Q31A',
}

# Build feature matrix
df_model = pd.DataFrame()
for label, col in feature_dict.items():
    if col in df.columns:
        df_model[label] = pd.to_numeric(df[col], errors='coerce')

# Build target
for label, col in target_cols.items():
    if col in df.columns:
        df_model[label] = df[col].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)

# Create composite mental health outcome
df_model['Mental_Health_Risk'] = ((df_model['Depression/Anxiety'] == 1) | 
                                   (df_model['Behavioral Problems'] == 1) | 
                                   (df_model['ADHD'] == 1)).astype(int)

# Filter valid data
df_model = df_model.dropna()
df_model = df_model[(df_model['Age'] >= 6) & (df_model['Screen Time'] <= 24)]

print(f"   Valid cases: {len(df_model):,}")
print(f"   Mental health risk prevalence: {df_model['Mental_Health_Risk'].mean() * 100:.1f}%")

# Prepare data for modeling
X = df_model[list(feature_dict.keys())]
y = df_model['Mental_Health_Risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n   Training set: {len(X_train):,} samples")
print(f"   Test set: {len(X_test):,} samples")

# MODEL 1: Logistic Regression
print("\n3. Training Logistic Regression model...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n   Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr, target_names=['No Risk', 'Risk']))
print(f"   AUC-ROC: {roc_auc_score(y_test, y_pred_proba_lr):.3f}")

# Get feature importance (coefficients)
lr_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\n   Feature Coefficients:")
print(lr_importance)

# MODEL 2: Random Forest
print("\n4. Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n   Random Forest Performance:")
print(classification_report(y_test, y_pred_rf, target_names=['No Risk', 'Risk']))
print(f"   AUC-ROC: {roc_auc_score(y_test, y_pred_proba_rf):.3f}")

# Get feature importance
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   Feature Importance:")
print(rf_importance)

# VISUALIZATION 1: Logistic Regression Coefficients
print("\n5. Generating Visualization 1: Logistic regression coefficients...")

fig, ax = plt.subplots(figsize=(36, 24))

colors = ['#e74c3c' if coef > 0 else '#3498db' for coef in lr_importance['Coefficient']]

bars = ax.barh(
    lr_importance['Feature'],
    lr_importance['Coefficient'],
    color=colors,
    edgecolor='black',
    linewidth=1.5,
    alpha=0.8
)

for bar, coef in zip(bars, lr_importance['Coefficient']):
    width = bar.get_width()
    ax.text(
        width + (0.02 if width > 0 else -0.02),
        bar.get_y() + bar.get_height()/2,
        f'{coef:.3f}',
        va='center',
        ha='left' if width > 0 else 'right',
        fontsize=14,
        fontweight='bold'
    )

ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax.set_xlabel('Coefficient (Standardized)', fontsize=18, fontweight='bold')
ax.set_ylabel('Feature', fontsize=18, fontweight='bold')
ax.set_title('Top Predictors of Mental Health Risk: Logistic Regression Coefficients\n'
             f'(NSCH 2023, n={len(df_model):,}, AUC={roc_auc_score(y_test, y_pred_proba_lr):.3f})',
             fontsize=22, fontweight='bold', pad=20)
ax.tick_params(axis='both', labelsize=16)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='Increases Risk', alpha=0.8),
    Patch(facecolor='#3498db', label='Decreases Risk', alpha=0.8)
]
ax.legend(handles=legend_elements, fontsize=14, loc='lower right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_predictors_logistic_regression.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/top_predictors_logistic_regression.png")

# VISUALIZATION 2: Random Forest Feature Importance
print("\n6. Generating Visualization 2: Random forest feature importance...")

fig, ax = plt.subplots(figsize=(36, 24))

colors_rf = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']

bars = ax.barh(
    rf_importance['Feature'],
    rf_importance['Importance'],
    color=colors_rf,
    edgecolor='black',
    linewidth=1.5,
    alpha=0.8
)

for bar, imp in zip(bars, rf_importance['Importance']):
    width = bar.get_width()
    ax.text(
        width + 0.005,
        bar.get_y() + bar.get_height()/2,
        f'{imp:.3f}',
        va='center',
        ha='left',
        fontsize=14,
        fontweight='bold'
    )

ax.set_xlabel('Feature Importance', fontsize=18, fontweight='bold')
ax.set_ylabel('Feature', fontsize=18, fontweight='bold')
ax.set_title('Top Predictors of Mental Health Risk: Random Forest Feature Importance\n'
             f'(NSCH 2023, n={len(df_model):,}, AUC={roc_auc_score(y_test, y_pred_proba_rf):.3f})',
             fontsize=22, fontweight='bold', pad=20)
ax.tick_params(axis='both', labelsize=16)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_predictors_random_forest.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/top_predictors_random_forest.png")

# VISUALIZATION 3: Model Comparison
print("\n7. Generating Visualization 3: Model comparison...")

fig, axes = plt.subplots(2, 2, figsize=(46, 34))

# Plot 1: Coefficient/Importance comparison
ax1 = axes[0, 0]

comparison_df = pd.DataFrame({
    'Feature': lr_importance['Feature'],
    'LR_Coefficient': lr_importance['Abs_Coefficient'].values,
    'RF_Importance': rf_importance.set_index('Feature').loc[lr_importance['Feature'], 'Importance'].values
})

x = np.arange(len(comparison_df))
width = 0.35

bars1 = ax1.bar(x - width/2, comparison_df['LR_Coefficient'], width, 
                label='Logistic Regression (|Coefficient|)', color='#3498db', 
                edgecolor='black', linewidth=1.2, alpha=0.8)
bars2 = ax1.bar(x + width/2, comparison_df['RF_Importance'], width,
                label='Random Forest (Importance)', color='#2ecc71',
                edgecolor='black', linewidth=1.2, alpha=0.8)

ax1.set_xlabel('Feature', fontsize=22, fontweight='bold')
ax1.set_ylabel('Importance Score', fontsize=22, fontweight='bold')
ax1.set_title('Model Comparison: Feature Importance', fontsize=24, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right', fontsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax1.legend(fontsize=16)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: AUC-ROC comparison
ax2 = axes[0, 1]

models = ['Logistic\nRegression', 'Random\nForest']
aucs = [roc_auc_score(y_test, y_pred_proba_lr), roc_auc_score(y_test, y_pred_proba_rf)]

bars = ax2.bar(models, aucs, color=['#3498db', '#2ecc71'], 
               edgecolor='black', linewidth=1.5, alpha=0.8)

for bar, auc in zip(bars, aucs):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f'{auc:.3f}', ha='center', va='bottom', 
             fontsize=16, fontweight='bold')

ax2.set_ylabel('AUC-ROC Score', fontsize=22, fontweight='bold')
ax2.set_title('Model Performance Comparison', fontsize=24, fontweight='bold', pad=15)
ax2.tick_params(axis='both', labelsize=18)
ax2.set_ylim([0, 1.1])
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance Level')
ax2.legend(fontsize=16)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Top 3 predictors visualization
ax3 = axes[1, 0]

top3_features = rf_importance['Feature'].head(3).tolist()
top3_colors = ['#e74c3c', '#9b59b6', '#2ecc71']

for i, (feature, color) in enumerate(zip(top3_features, top3_colors)):
    feature_data = df_model[df_model['Mental_Health_Risk'] == 0][feature]
    risk_data = df_model[df_model['Mental_Health_Risk'] == 1][feature]
    
    positions = [i*2, i*2 + 0.8]
    bp = ax3.boxplot([feature_data, risk_data], positions=positions, 
                      widths=0.6, patch_artist=True,
                      labels=['No Risk', 'Risk'])
    
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.text(i*2 + 0.4, ax3.get_ylim()[1] * 0.95, feature, 
             ha='center', fontsize=16, fontweight='bold')

ax3.set_ylabel('Feature Value', fontsize=22, fontweight='bold')
ax3.set_title('Top 3 Predictors: Distribution by Risk Status', fontsize=24, fontweight='bold', pad=15)
ax3.tick_params(axis='both', labelsize=18)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Feature ranking table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

table_data = [['Rank', 'Feature', 'LR Coef.', 'RF Import.', 'Direction']]
for rank, (idx, row) in enumerate(lr_importance.head(5).iterrows(), 1):
    feature = row['Feature']
    lr_coef = row['Coefficient']
    rf_imp = rf_importance[rf_importance['Feature'] == feature]['Importance'].values[0]
    direction = '↑ Risk' if lr_coef > 0 else '↓ Risk'
    table_data.append([str(rank), feature, f"{lr_coef:.3f}", f"{rf_imp:.3f}", direction])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.1, 0.3, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 3)

for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=18)

for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#f8f9fa')
        else:
            cell.set_facecolor('#ffffff')

ax4.set_title('Feature Rankings Summary', fontsize=24, fontweight='bold', pad=20)

plt.suptitle('Comprehensive Predictive Analysis: Mental Health Risk Factors (NSCH 2023)',
             fontsize=24, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f"{OUTPUT_DIR}/top_predictors_comprehensive.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/top_predictors_comprehensive.png")

# Summary
print("\n" + "="*80)
print("PREDICTIVE MODELING COMPLETE - KEY FINDINGS:")
print("="*80)

print("\nLogistic Regression Top 3 Predictors:")
for idx, row in lr_importance.head(3).iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {row['Feature']}: coefficient = {row['Coefficient']:.3f} ({direction} risk)")

print("\nRandom Forest Top 3 Predictors:")
for idx, row in rf_importance.head(3).iterrows():
    print(f"  {row['Feature']}: importance = {row['Importance']:.3f}")

print(f"\nModel Performance:")
print(f"  - Logistic Regression AUC-ROC: {roc_auc_score(y_test, y_pred_proba_lr):.3f}")
print(f"  - Random Forest AUC-ROC: {roc_auc_score(y_test, y_pred_proba_rf):.3f}")

print("\n" + "="*80)
print("Generated 3 predictive modeling visualizations:")
print("  1. top_predictors_logistic_regression.png")
print("  2. top_predictors_random_forest.png")
print("  3. top_predictors_comprehensive.png")
print("="*80)
