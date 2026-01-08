"""
Screen Time Categorical Analysis with Mental Health Outcomes (NSCH 2023)
High-impact analysis showing relationship between screen time and mental health
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = "data/nsch_2023e_topical.csv"
OUTPUT_DIR = "visualizations"

print("="*80)
print("SCREEN TIME & MENTAL HEALTH ANALYSIS (NSCH 2023)")
print("="*80)

# Load data
print("\n1. Loading NSCH data...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded: {len(df):,} children")

# Extract key variables
print("\n2. Extracting variables...")

# Screen time (hours per weekday)
df['screen_time'] = pd.to_numeric(df['SCREENTIME'], errors='coerce')

# Mental health indicators (1=Yes, 2=No typically in NSCH)
# Depression/anxiety
df['depression_anxiety'] = df['K2Q32A'].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)

# Behavioral/conduct problems
df['behavior_problem'] = df['K2Q33A'].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)

# ADHD
df['adhd'] = df['K2Q31A'].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)

# Autism
df['autism'] = df['K2Q35A'].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)

# Any mental health condition
df['any_mental_health'] = ((df['depression_anxiety'] == 1) | 
                            (df['behavior_problem'] == 1) | 
                            (df['adhd'] == 1)).astype(int)

# Sleep hours
df['sleep_hours'] = pd.to_numeric(df['HOURSLEEP'], errors='coerce')

# Age
df['age'] = pd.to_numeric(df['SC_AGE_YEARS'], errors='coerce')

# Filter to valid data
df_clean = df[
    (df['screen_time'].notna()) & 
    (df['age'].notna()) &
    (df['age'] >= 6) &  # School-age children
    (df['screen_time'] <= 24)  # Valid hours
].copy()

print(f"   Valid cases: {len(df_clean):,}")

# Create screen time categories
print("\n3. Creating screen time categories...")
df_clean['screen_category'] = pd.cut(
    df_clean['screen_time'],
    bins=[0, 2, 4, 6, 24],
    labels=['<2 hours', '2-4 hours', '4-6 hours', '6+ hours'],
    include_lowest=True
)

# Summary statistics by category
print("\n4. Computing statistics by screen time category...")
summary = df_clean.groupby('screen_category').agg({
    'depression_anxiety': 'mean',
    'behavior_problem': 'mean',
    'adhd': 'mean',
    'any_mental_health': 'mean',
    'sleep_hours': 'mean',
    'age': ['mean', 'count']
}).round(3)

print(summary)

# VISUALIZATION 1: Mental Health Prevalence by Screen Time Category
print("\n5. Generating Visualization 1: Mental health prevalence by screen time...")

fig, axes = plt.subplots(2, 2, figsize=(32, 24))
axes = axes.flatten()

conditions = [
    ('depression_anxiety', 'Depression/Anxiety', '#e74c3c'),
    ('behavior_problem', 'Behavioral Problems', '#3498db'),
    ('adhd', 'ADHD', '#2ecc71'),
    ('any_mental_health', 'Any Mental Health Condition', '#f39c12')
]

for idx, (col, title, color) in enumerate(conditions):
    ax = axes[idx]
    
    # Calculate prevalence by category
    prevalence = df_clean.groupby('screen_category')[col].agg(['mean', 'count']).reset_index()
    prevalence['prevalence_pct'] = prevalence['mean'] * 100
    
    # Create bar chart
    bars = ax.bar(
        range(len(prevalence)),
        prevalence['prevalence_pct'],
        color=color,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add value labels
    for bar, val, n in zip(bars, prevalence['prevalence_pct'], prevalence['count']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f'{val:.1f}%\n(n={n:,})',
            ha='center',
            va='bottom',
            fontsize=28,
            fontweight='bold'
        )
    
    ax.set_xlabel('Screen Time Category', fontsize=42, fontweight='bold')
    ax.set_ylabel('Prevalence (%)', fontsize=42, fontweight='bold')
    ax.set_title(f'{title}', fontsize=44, fontweight='bold', pad=30)
    ax.set_xticks(range(len(prevalence)))
    ax.set_xticklabels(prevalence['screen_category'], fontsize=36)
    ax.tick_params(axis='y', labelsize=36)
    ax.set_ylim([0, max(prevalence['prevalence_pct']) * 1.3])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Mental Health Prevalence by Screen Time Category (NSCH 2023, Ages 6+)', 
             fontsize=48, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f"{OUTPUT_DIR}/screentime_mental_health_prevalence.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/screentime_mental_health_prevalence.png")

# VISUALIZATION 2: Dose-Response Relationship
print("\n6. Generating Visualization 2: Dose-response curves...")

fig, axes = plt.subplots(1, 3, figsize=(50, 20))

# Plot 1: Depression/Anxiety by screen time bins
screen_bins = np.arange(0, 13, 1)
prevalence_by_hour = []
for i in range(len(screen_bins)-1):
    mask = (df_clean['screen_time'] >= screen_bins[i]) & (df_clean['screen_time'] < screen_bins[i+1])
    if mask.sum() > 10:
        prev = df_clean[mask]['depression_anxiety'].mean() * 100
        prevalence_by_hour.append((screen_bins[i] + 0.5, prev, mask.sum()))

if prevalence_by_hour:
    hours, prev, counts = zip(*prevalence_by_hour)
    axes[0].plot(hours, prev, 'o-', color='#e74c3c', linewidth=6, markersize=20)
    axes[0].set_xlabel('Daily Screen Time (hours)', fontsize=65, fontweight='bold')
    axes[0].set_ylabel('Depression/Anxiety Prevalence (%)', fontsize=65, fontweight='bold')
    axes[0].set_title('Depression/Anxiety vs Screen Time', fontsize=70, fontweight='bold', pad=40)
    axes[0].tick_params(axis='both', labelsize=58)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 12])

# Plot 2: Sleep hours by screen time category
sleep_data = df_clean.groupby('screen_category')['sleep_hours'].agg(['mean', 'std', 'count']).reset_index()
sleep_data = sleep_data[sleep_data['count'] > 10]

bars = axes[1].bar(
    range(len(sleep_data)),
    sleep_data['mean'],
    yerr=sleep_data['std'] / np.sqrt(sleep_data['count']),
    color='#3498db',
    edgecolor='black',
    linewidth=1.5,
    alpha=0.8,
    capsize=5
)

for bar, val in zip(bars, sleep_data['mean']):
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width()/2,
        height + 0.1,
        f'{val:.1f}h',
        ha='center',
        va='bottom',
        fontsize=44,
        fontweight='bold'
    )

axes[1].set_xlabel('Screen Time Category', fontsize=65, fontweight='bold')
axes[1].set_ylabel('Average Sleep Hours', fontsize=65, fontweight='bold')
axes[1].set_title('Sleep Duration vs Screen Time', fontsize=70, fontweight='bold', pad=40)
axes[1].set_xticks(range(len(sleep_data)))
axes[1].set_xticklabels(sleep_data['screen_category'], fontsize=58)
axes[1].tick_params(axis='y', labelsize=58)
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(y=9, color='red', linestyle='--', linewidth=5, label='Recommended (9h)')
axes[1].legend(fontsize=44)

# Plot 3: Multiple conditions stacked
conditions_stack = ['depression_anxiety', 'behavior_problem', 'adhd']
condition_labels = ['Depression/Anxiety', 'Behavioral Problems', 'ADHD']
colors_stack = ['#e74c3c', '#3498db', '#2ecc71']

stack_data = df_clean.groupby('screen_category')[conditions_stack].mean() * 100
x_pos = np.arange(len(stack_data))

bottom = np.zeros(len(stack_data))
for i, (cond, label, color) in enumerate(zip(conditions_stack, condition_labels, colors_stack)):
    axes[2].bar(
        x_pos,
        stack_data[cond],
        bottom=bottom,
        label=label,
        color=color,
        edgecolor='black',
        linewidth=1,
        alpha=0.8
    )
    bottom += stack_data[cond]

axes[2].set_xlabel('Screen Time Category', fontsize=65, fontweight='bold')
axes[2].set_ylabel('Cumulative Prevalence (%)', fontsize=65, fontweight='bold')
axes[2].set_title('Mental Health Conditions (Stacked)', fontsize=70, fontweight='bold', pad=40)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(stack_data.index, fontsize=58)
axes[2].tick_params(axis='y', labelsize=58)
axes[2].legend(fontsize=46, loc='upper left')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Screen Time & Mental Health: Dose-Response Analysis (NSCH 2023)', 
             fontsize=75, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/screentime_dose_response.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/screentime_dose_response.png")

# VISUALIZATION 3: Statistical Testing
print("\n7. Generating Visualization 3: Statistical comparison...")

fig, ax = plt.subplots(figsize=(28, 18))

# Prepare data for statistical testing
categories = df_clean['screen_category'].unique()
categories = sorted([c for c in categories if pd.notna(c)])

test_results = []
for condition, label, _ in conditions:
    row = [label]
    baseline_data = df_clean[df_clean['screen_category'] == '<2 hours'][condition].dropna()
    
    for cat in categories:
        cat_data = df_clean[df_clean['screen_category'] == cat][condition].dropna()
        if len(cat_data) > 10 and len(baseline_data) > 10:
            # Chi-square test
            contingency = pd.crosstab(
                df_clean[df_clean['screen_category'].isin(['<2 hours', cat])]['screen_category'],
                df_clean[df_clean['screen_category'].isin(['<2 hours', cat])][condition]
            )
            if contingency.shape == (2, 2) and contingency.values.min() >= 5:
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                prev = cat_data.mean() * 100
                
                if cat == '<2 hours':
                    row.append(f"{prev:.1f}%\n(ref)")
                elif p_value < 0.001:
                    row.append(f"{prev:.1f}%\n***")
                elif p_value < 0.01:
                    row.append(f"{prev:.1f}%\n**")
                elif p_value < 0.05:
                    row.append(f"{prev:.1f}%\n*")
                else:
                    row.append(f"{prev:.1f}%\nns")
            else:
                row.append("N/A")
        else:
            row.append("N/A")
    
    test_results.append(row)

# Create table
table_data = [['Condition'] + [str(c) for c in categories]] + test_results

ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    colWidths=[0.30] + [0.175] * len(categories)
)

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 3.5)

# Style header
for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=18)

# Style first column
for i in range(1, len(table_data)):
    cell = table[(i, 0)]
    cell.set_facecolor('#ecf0f1')
    cell.set_text_props(weight='bold', fontsize=16)

# Style data cells
for i in range(1, len(table_data)):
    for j in range(1, len(table_data[0])):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ffffff')
        else:
            cell.set_facecolor('#f8f9fa')

ax.text(
    0.5, -0.08,
    'Statistical significance: * p<0.05, ** p<0.01, *** p<0.001, ns=not significant\n'
    'Comparison to <2 hours reference group using Chi-square tests',
    transform=ax.transAxes,
    ha='center',
    fontsize=16,
    style='italic',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)

ax.set_title('Mental Health Prevalence by Screen Time with Statistical Testing (NSCH 2023)', 
             fontsize=24, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/screentime_statistical_testing.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/screentime_statistical_testing.png")

# Summary statistics
print("\n" + "="*80)
print("ANALYSIS COMPLETE - KEY FINDINGS:")
print("="*80)

print("\nPrevalence by Screen Time Category:")
for cat in categories:
    cat_df = df_clean[df_clean['screen_category'] == cat]
    n = len(cat_df)
    dep_pct = cat_df['depression_anxiety'].mean() * 100
    beh_pct = cat_df['behavior_problem'].mean() * 100
    any_pct = cat_df['any_mental_health'].mean() * 100
    sleep_avg = cat_df['sleep_hours'].mean()
    
    print(f"\n{cat} (n={n:,}):")
    print(f"  - Depression/Anxiety: {dep_pct:.1f}%")
    print(f"  - Behavioral Problems: {beh_pct:.1f}%")
    print(f"  - Any Mental Health: {any_pct:.1f}%")
    print(f"  - Average Sleep: {sleep_avg:.1f} hours")

print("\n" + "="*80)
print("Generated 3 high-impact visualizations:")
print("  1. screentime_mental_health_prevalence.png")
print("  2. screentime_dose_response.png")
print("  3. screentime_statistical_testing.png")
print("="*80)
