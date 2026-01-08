"""
Generate SHAP Dependence Plots for IGD (Internet Gaming Disorder) Dataset
Shows how each feature affects model predictions for IGD risk
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
    "figure.figsize": (16, 12),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 22,
    "axes.titlesize": 32,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
})

# Create output directory
OUTPUT_DIR = "visualizations/igd_shap_dependence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING SHAP DEPENDENCE PLOTS FOR IGD DATASET")
print("=" * 80)

# Load data
print("\n1. Loading IGD data...")
DATA_PATH = "data/IGD Database.xlsx"
df = pd.read_excel(DATA_PATH)
print(f"   Loaded {df.shape[0]:,} records with {df.shape[1]} features")

# Convert categorical variables to numeric
print("\n2. Preprocessing data...")
weekday_hours_map = {'1 or less': 1, '2 to 3': 2.5, '4 to 5': 4.5, '6 to 7': 6.5, '8 to 10': 9}
weekend_hours_map = {'1 or less': 1, '2 to 3': 2.5, '4 to 5': 4.5, '6 to 7': 6.5, '8 to 10': 9, '11 or more': 11}

df['Weekday Hours'] = df['Weekday Hours'].map(weekday_hours_map)
df['Weekend Hours'] = df['Weekend Hours'].map(weekend_hours_map)

# Feature names and labels
feature_names = {
    'Weekday Hours': 'Weekday Gaming Hours',
    'Weekend Hours': 'Weekend Gaming Hours',
    'Sleep Quality': 'Sleep Quality Score',
    'IGD Total': 'IGD Total Score',
    'Social': 'Social Gaming Score',
    'Escape': 'Gaming as Escape Score'
}

features = list(feature_names.keys())
target = 'IGD Status'

# Prepare data
print("\n3. Preparing features and target...")
df_clean = df.dropna(subset=features + [target])
X = df_clean[features].copy()
y = (df_clean[target] == 'Y').astype(int)

print(f"   Features: {len(features)}")
print(f"   IGD Positive cases: {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"   IGD Negative cases: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")

# Split data
print("\n4. Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train):,} samples (IGD+: {y_train.sum()}, IGD-: {(y_train==0).sum()})")
print(f"   Test:  {len(X_test):,} samples (IGD+: {y_test.sum()}, IGD-: {(y_test==0).sum()})")

# Train model with class balancing
print("\n5. Training Random Forest model...")
# Calculate class weight for imbalance
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_weight = n_neg / n_pos

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"   Train Accuracy: {train_score:.3f}")
print(f"   Test Accuracy:  {test_score:.3f}")
print(f"   Class weight ratio (N:Y): {scale_weight:.2f}:1")

# Calculate SHAP values
print("\n6. Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_test)

# For binary classification
if isinstance(shap_values_raw, list):
    shap_values = shap_values_raw[1]  # Use positive class (IGD positive)
elif len(shap_values_raw.shape) == 3:
    shap_values = shap_values_raw[:, :, 1]  # Use positive class
else:
    shap_values = shap_values_raw

print(f"   SHAP values shape: {shap_values.shape}")

# Generate SHAP summary plot
print("\n7. Generating SHAP summary plot...")
fig, ax = plt.subplots(figsize=(18, 12))
shap.summary_plot(
    shap_values, 
    X_test,
    feature_names=[feature_names[f] for f in features],
    show=False,
    max_display=len(features)
)
plt.title("Feature Impact Distribution", 
          fontsize=12, fontweight='bold', pad=15)
plt.xlabel("SHAP Value (impact on model output)", fontsize=12, fontweight='bold')
plt.ylabel("", fontsize=12, fontweight='bold')  # Empty ylabel, features are already labeled
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# Adjust colorbar font size
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=10)
cbar.set_ylabel('Feature Value', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/00_shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 00_shap_summary_plot.png")

# Generate SHAP dependence plots for each feature
print("\n8. Generating SHAP dependence plots for each feature...")
print("   " + "-" * 60)

for idx, feature in enumerate(features, 1):
    print(f"   {idx}/{len(features)}: {feature_names[feature]}")
    
    # Create dependence plot with larger figure
    fig, ax = plt.subplots(figsize=(18, 12))
    
    shap.dependence_plot(
        idx - 1,  # Use feature index
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
        fontsize=24,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel(feature_names[feature], fontsize=20, fontweight='bold')
    ax.set_ylabel(
        f"SHAP value for {feature_names[feature]}",
        fontsize=20,
        fontweight='bold'
    )
    
    # Make ticks larger
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Adjust colorbar font size if present
    if hasattr(ax, 'collections') and len(ax.collections) > 0:
        for collection in ax.collections:
            if hasattr(collection, 'colorbar') and collection.colorbar is not None:
                collection.colorbar.ax.tick_params(labelsize=18)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save with zero-padded numbering
    filename = f"{OUTPUT_DIR}/{idx:02d}_dependence_{feature.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

print("   " + "-" * 60)

# Generate bar plot of mean absolute SHAP values (Feature Importance)
print("\n9. Generating feature importance (mean |SHAP|) plot...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': [feature_names[f] for f in features],
    'Mean |SHAP|': mean_abs_shap
}).sort_values('Mean |SHAP|', ascending=True)

fig, ax = plt.subplots(figsize=(16, 10))
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
        val + 0.002,
        bar.get_y() + bar.get_height()/2,
        f'{val:.4f}',
        va='center',
        fontsize=14,
        fontweight='bold'
    )

ax.set_xlabel('Mean Absolute SHAP Value', fontsize=18, fontweight='bold')
ax.set_ylabel('Feature', fontsize=18, fontweight='bold')
ax.set_title(
    'Feature Importance Ranking for IGD Prediction',
    fontsize=22,
    fontweight='bold',
    pad=15
)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/00_feature_importance_bar.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 00_feature_importance_bar.png")

# Generate maximum SHAP impact plot
print("\n10. Generating maximum SHAP impact by feature plot...")
max_abs_shap = np.abs(shap_values).max(axis=0)
max_impact_df = pd.DataFrame({
    'Feature': [feature_names[f] for f in features],
    'Max |SHAP|': max_abs_shap
}).sort_values('Max |SHAP|', ascending=True)

fig, ax = plt.subplots(figsize=(16, 10))
bars = ax.barh(
    max_impact_df['Feature'], 
    max_impact_df['Max |SHAP|'],
    color='coral',
    edgecolor='black',
    linewidth=1.5
)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, max_impact_df['Max |SHAP|'])):
    ax.text(
        val + 0.005,
        bar.get_y() + bar.get_height()/2,
        f'{val:.4f}',
        va='center',
        fontsize=14,
        fontweight='bold'
    )

ax.set_xlabel('Maximum Absolute SHAP Value', fontsize=18, fontweight='bold')
ax.set_ylabel('Feature', fontsize=18, fontweight='bold')
ax.set_title(
    'Maximum SHAP Impact by Feature (IGD Dataset)',
    fontsize=22,
    fontweight='bold',
    pad=15
)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/00_maximum_shap_impact.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 00_maximum_shap_impact.png")

# Create interaction heatmap
print("\n11. Generating SHAP interaction heatmap...")
try:
    # Compute pairwise correlation of SHAP values as proxy for interactions
    fig, ax = plt.subplots(figsize=(14, 12))
    
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
        annot_kws={'fontsize': 14},
        ax=ax
    )
    
    ax.set_title(
        'Feature Interaction Patterns (IGD Dataset)',
        fontsize=22,
        fontweight='bold',
        pad=20
    )
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    
    # Update colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label("Absolute Correlation of SHAP Values", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_interaction_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 11_interaction_heatmap.png")
except Exception as e:
    print(f"   ⚠ Could not generate interaction heatmap: {e}")

# Create summary statistics table
print("\n12. Generating summary statistics...")
summary_stats = pd.DataFrame({
    'Feature': [feature_names[f] for f in features],
    'Mean SHAP': shap_values.mean(axis=0),
    'Std SHAP': shap_values.std(axis=0),
    'Mean |SHAP|': np.abs(shap_values).mean(axis=0),
    'Max |SHAP|': np.abs(shap_values).max(axis=0)
}).sort_values('Mean |SHAP|', ascending=False)

print("\n" + "=" * 100)
print("SHAP VALUE STATISTICS FOR IGD DATASET")
print("=" * 100)
print(summary_stats.to_string(index=False))
print("=" * 100)

# Save summary to CSV
summary_path = f"{OUTPUT_DIR}/shap_statistics_igd.csv"
summary_stats.to_csv(summary_path, index=False)
print(f"\n✓ Summary statistics saved to: {summary_path}")

# Final summary
print("\n" + "=" * 80)
print("✅ IGD SHAP DEPENDENCE PLOTS GENERATION COMPLETE!")
print("=" * 80)
print(f"\nGenerated files in '{OUTPUT_DIR}/':")
print(f"  • 00_shap_summary_plot.png       - Overall SHAP summary")
print(f"  • 00_feature_importance_bar.png  - Feature importance ranking")
print(f"  • 00_maximum_shap_impact.png     - Maximum SHAP impact by feature")
print(f"  • 01-06_dependence_*.png         - Individual feature dependence plots")
print(f"  • 11_interaction_heatmap.png     - Feature interaction patterns")
print(f"  • shap_statistics_igd.csv        - Numerical statistics")
print("\nKey Insights:")
print(f"  • Top 3 most important features:")
for i, (idx, row) in enumerate(summary_stats.head(3).iterrows(), 1):
    print(f"    {i}. {row['Feature']}: Mean |SHAP| = {row['Mean |SHAP|']:.4f}")
print(f"\n  • Model Performance:")
print(f"    - Train Accuracy: {train_score:.1%}")
print(f"    - Test Accuracy:  {test_score:.1%}")
print(f"    - IGD Positive Prevalence: {y.mean():.1%}")
print(f"    - Class Imbalance Ratio: {scale_weight:.2f}:1 (N:Y)")
print("=" * 80)
