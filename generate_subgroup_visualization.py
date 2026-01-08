"""
Generate Subgroup Analysis Visualization for IGD Dataset
Analyzes model performance by sex subgroups
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('default')
sns.set_palette("husl")

OUTPUT_DIR = "visualizations"

print("="*80)
print("GENERATING SUBGROUP ANALYSIS VISUALIZATION")
print("="*80)

# Load the subgroup analysis results
df = pd.read_csv('visualizations/subgroup_analysis_sex.csv')

print("\n1. Loading subgroup analysis results...")
print(f"   Total rows: {len(df)}")
print(f"   Models: {df['Model'].nunique()}")
print(f"   Subgroups: {df['Sex'].unique()}")

# Filter for male subgroup only (female has 0 IGD+ cases)
df_male = df[df['Sex'] == 'Male'].copy()

print("\n2. Data Summary:")
print(f"   Male subgroup: N={df_male.iloc[0]['N']}, IGD+={df_male.iloc[0]['IGD+']}")
print(f"   Female subgroup: N={df[df['Sex']=='Female'].iloc[0]['N']}, IGD+=0 (no visualization)")

# Create visualization
print("\n3. Generating subgroup performance visualization...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Create bar chart
    bars = ax.barh(
        df_male['Model'],
        df_male[metric],
        color=colors[idx],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.02,
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            va='center',
            fontsize=14,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_xlabel(metric, fontsize=18, fontweight='bold')
    ax.set_ylabel('Model', fontsize=18, fontweight='bold')
    ax.set_title(f'{metric} by Model (Male Subgroup)', 
                 fontsize=20, fontweight='bold', pad=15)
    ax.set_xlim([0, 1.1])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add overall title
fig.suptitle('Subgroup Analysis: Model Performance by Sex (IGD Dataset)\nMale Subgroup Only (N=35, IGD+=3)', 
             fontsize=22, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f"{OUTPUT_DIR}/subgroup_analysis_visualization.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/subgroup_analysis_visualization.png")

# Create a summary comparison table visualization
print("\n4. Generating summary comparison table...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Note'])

for _, row in df_male.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Accuracy']:.3f}",
        f"{row['Precision']:.3f}",
        f"{row['Recall']:.3f}",
        f"{row['F1']:.3f}",
        'Male only'
    ])

# Add female note
table_data.append(['All Models', 'N/A', 'N/A', 'N/A', 'N/A', 'Female: 0 IGD+ cases'])

# Create table
table = ax.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.15]
)

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 3)

# Style header row
for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=16)

# Style data rows
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == len(table_data) - 1:  # Last row (female note)
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(style='italic', fontsize=13)
        else:
            if i % 2 == 0:
                cell.set_facecolor('#ffffff')
            else:
                cell.set_facecolor('#f8f9fa')

ax.set_title('Subgroup Analysis Summary: Performance by Sex (IGD Dataset)\nTest Set: Male N=35 (3 IGD+), Female N=27 (0 IGD+)', 
             fontsize=22, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/subgroup_analysis_table.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/subgroup_analysis_table.png")

# Create a simple comparison chart
print("\n5. Generating model comparison chart...")

fig, ax = plt.subplots(figsize=(20, 12))

models = df_male['Model'].tolist()
x = np.arange(len(models))
width = 0.18

bars1 = ax.bar(x - 1.5*width, df_male['Accuracy'], width, label='Accuracy', color='#3498db', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x - 0.5*width, df_male['Precision'], width, label='Precision', color='#e74c3c', edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + 0.5*width, df_male['Recall'], width, label='Recall', color='#2ecc71', edgecolor='black', linewidth=1.2)
bars4 = ax.bar(x + 1.5*width, df_male['F1'], width, label='F1-Score', color='#f39c12', edgecolor='black', linewidth=1.2)

# Add value labels with rotation to prevent overlap
def add_labels(bars, values):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=0)

add_labels(bars1, df_male['Accuracy'])
add_labels(bars2, df_male['Precision'])
add_labels(bars3, df_male['Recall'])
add_labels(bars4, df_male['F1'])

ax.set_xlabel('Model', fontsize=18, fontweight='bold', labelpad=10)
ax.set_ylabel('Score', fontsize=18, fontweight='bold')
ax.set_title('Subgroup Analysis: Model Performance Metrics (Male Subgroup, N=35, IGD+=3)', 
             fontsize=22, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0, ha='center', fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16, pad=8)
ax.legend(fontsize=16, loc='upper left', frameon=True, shadow=True, ncol=4)
ax.set_ylim([0, 1.2])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add note about female subgroup
ax.text(0.5, -0.15, 'Note: Female subgroup (N=27) had 0 IGD+ cases, precluding sensitivity analysis', 
        transform=ax.transAxes, ha='center', fontsize=14, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/subgroup_comparison_chart.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/subgroup_comparison_chart.png")

print("\n" + "="*80)
print("[SUCCESS] SUBGROUP ANALYSIS VISUALIZATION COMPLETE!")
print("="*80)
print("\nGenerated files in 'visualizations/':")
print("  • subgroup_analysis_visualization.png - 4-panel metric comparison")
print("  • subgroup_analysis_table.png         - Summary table")
print("  • subgroup_comparison_chart.png       - Grouped bar chart")
print("\nKey Findings:")
print("  • All IGD+ cases in test set were male (3/35)")
print("  • Female subgroup had 0 IGD+ cases (0/27)")
print("  • Subgroup analysis limited by low positive case count")
print("  • Models achieved 91-94% accuracy on male subgroup")
print("  • Recall ranged from 33% (MLP) to 100% (multiple models)")
print("="*80)
