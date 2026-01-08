"""
Generate SHAP Dependence Plots for NSCH Mental Health Risk Model
Shows how each feature affects model predictions across its value range
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (14, 10),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 16,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# Create output directory
OUTPUT_DIR = "visualizations/shap_dependence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING SHAP DEPENDENCE PLOTS")
print("=" * 80)

# Load data
print("\n1. Loading NSCH data...")
DATA_PATH = "data/nsch_2023e_topical.csv"
df = pd.read_csv(DATA_PATH)
print(f"   Loaded {df.shape[0]:,} records with {df.shape[1]} features")

# Feature selection - key predictors
feature_names = {
    'SCREENTIME': 'Screen Time (hours)',
    'A1_MENTHEALTH': 'Mental Health Status',
    'K2Q31A': 'Anxiety Problems',
    'K2Q32A': 'Depression',
    'K2Q33A': 'Behavior Problems',
    'HOURSLEEP': 'Sleep Hours',
    'PHYSACTIV': 'Physical Activity',
    'K7Q70_R': 'Emotional Support',
    'PLAYWELL': 'Social Behavior'
}

features = list(feature_names.keys())

# Prepare data
print("\n2. Preparing features and target...")
X = df[features].copy()

# Handle missing values
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col].fillna(X[col].mean(), inplace=True)

# Define high-risk target (poor mental health indicators)
y = ((df['A1_MENTHEALTH'] <= 2) | 
     (df['K2Q31A'] == 1) | 
     (df['K2Q32A'] == 1) | 
     (df['K2Q33A'] == 1)).astype(int)

print(f"   Features: {len(features)}")
print(f"   High-risk cases: {y.sum():,} ({y.mean()*100:.1f}%)")

# Split data
print("\n3. Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train):,} samples")
print(f"   Test:  {len(X_test):,} samples")

# Train model
print("\n4. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"   Train Accuracy: {train_score:.3f}")
print(f"   Test Accuracy:  {test_score:.3f}")

# Calculate SHAP values
print("\n5. Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_test)

# For binary classification, shap_values might be a list or 3D array
if isinstance(shap_values_raw, list):
    shap_values = shap_values_raw[1]  # Use positive class (high-risk)
elif len(shap_values_raw.shape) == 3:
    shap_values = shap_values_raw[:, :, 1]  # Use positive class
else:
    shap_values = shap_values_raw

print(f"   SHAP values shape: {shap_values.shape}")

# Generate SHAP summary plot
print("\n6. Generating SHAP summary plot...")
fig, ax = plt.subplots(figsize=(14, 10))
shap.summary_plot(
    shap_values, 
    X_test,
    feature_names=[feature_names[f] for f in features],
    show=False,
    max_display=len(features)
)
plt.title("SHAP Summary: Feature Impact on Mental Health Risk Predictions", 
          fontsize=24, fontweight='bold', pad=20)
plt.xlabel("SHAP Value (impact on model output)", fontsize=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/00_shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 00_shap_summary_plot.png")

# Generate SHAP dependence plots for each feature
print("\n7. Generating SHAP dependence plots for each feature...")
print("   " + "-" * 60)

for idx, feature in enumerate(features, 1):
    print(f"   {idx}/{len(features)}: {feature_names[feature]}")
    
    # Create dependence plot
    fig, ax = plt.subplots(figsize=(14, 9))
    
    shap.dependence_plot(
        idx - 1,  # Use feature index instead of name
        shap_values,
        X_test,
        feature_names=[feature_names[f] for f in features],
        interaction_index="auto",  # Automatically select best interaction
        show=False,
        ax=ax
    )
    
    # Enhance the plot
    ax.set_title(
        f"SHAP Dependence: {feature_names[feature]}",
        fontsize=26,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel(feature_names[feature], fontsize=22, fontweight='bold')
    ax.set_ylabel(
        f"SHAP value for {feature_names[feature]}",
        fontsize=22,
        fontweight='bold'
    )
    
    # Make ticks larger
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save with zero-padded numbering
    filename = f"{OUTPUT_DIR}/{idx:02d}_dependence_{feature.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("   " + "-" * 60)

# Generate bar plot of mean absolute SHAP values
print("\n8. Generating feature importance (mean |SHAP|) plot...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': [feature_names[f] for f in features],
    'Mean |SHAP|': mean_abs_shap
}).sort_values('Mean |SHAP|', ascending=True)

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(
    feature_importance_df['Feature'], 
    feature_importance_df['Mean |SHAP|'],
    color='steelblue',
    edgecolor='black',
    linewidth=1.5
)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, feature_importance_df['Mean |SHAP|'])):
    ax.text(
        val + 0.001,
        bar.get_y() + bar.get_height()/2,
        f'{val:.4f}',
        va='center',
        fontsize=16,
        fontweight='bold'
    )

ax.set_xlabel('Mean Absolute SHAP Value', fontsize=22, fontweight='bold')
ax.set_ylabel('Feature', fontsize=22, fontweight='bold')
ax.set_title(
    'Feature Importance: Mean Absolute Impact on Predictions',
    fontsize=26,
    fontweight='bold',
    pad=20
)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/00_feature_importance_bar.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 00_feature_importance_bar.png")

# Create interaction heatmap
print("\n9. Generating SHAP interaction heatmap...")
try:
    # Calculate average absolute interaction effects
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Compute pairwise correlation of SHAP values as proxy for interactions
    shap_df = pd.DataFrame(shap_values, columns=features)
    interaction_matrix = shap_df.corr().abs()
    
    sns.heatmap(
        interaction_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        center=0.5,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Absolute Correlation of SHAP Values"},
        xticklabels=[feature_names[f] for f in features],
        yticklabels=[feature_names[f] for f in features],
        ax=ax
    )
    
    ax.set_title(
        'Feature Interaction Patterns (SHAP Value Correlations)',
        fontsize=26,
        fontweight='bold',
        pad=20
    )
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_interaction_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 10_interaction_heatmap.png")
except Exception as e:
    print(f"   ⚠ Could not generate interaction heatmap: {e}")

# Create summary statistics table
print("\n10. Generating summary statistics...")
summary_stats = pd.DataFrame({
    'Feature': [feature_names[f] for f in features],
    'Mean SHAP': shap_values.mean(axis=0),
    'Std SHAP': shap_values.std(axis=0),
    'Mean |SHAP|': np.abs(shap_values).mean(axis=0),
    'Max |SHAP|': np.abs(shap_values).max(axis=0)
}).sort_values('Mean |SHAP|', ascending=False)

print("\n" + "=" * 90)
print("SHAP VALUE STATISTICS")
print("=" * 90)
print(summary_stats.to_string(index=False))
print("=" * 90)

# Save summary to CSV
summary_path = f"{OUTPUT_DIR}/shap_statistics.csv"
summary_stats.to_csv(summary_path, index=False)
print(f"\n✓ Summary statistics saved to: {summary_path}")

# Final summary
print("\n" + "=" * 80)
print("✅ SHAP DEPENDENCE PLOTS GENERATION COMPLETE!")
print("=" * 80)
print(f"\nGenerated files in '{OUTPUT_DIR}/':")
print(f"  • 00_shap_summary_plot.png       - Overall SHAP summary")
print(f"  • 00_feature_importance_bar.png  - Feature importance ranking")
print(f"  • 01-09_dependence_*.png         - Individual feature dependence plots")
print(f"  • 10_interaction_heatmap.png     - Feature interaction patterns")
print(f"  • shap_statistics.csv            - Numerical statistics")
print("\nKey Insights:")
print(f"  • Top 3 most important features:")
for i, row in summary_stats.head(3).iterrows():
    print(f"    {i+1}. {row['Feature']}: Mean |SHAP| = {row['Mean |SHAP|']:.4f}")
print(f"\n  • Model Performance:")
print(f"    - Train Accuracy: {train_score:.1%}")
print(f"    - Test Accuracy:  {test_score:.1%}")
print(f"    - High-risk Prevalence: {y.mean():.1%}")
print("=" * 80)
