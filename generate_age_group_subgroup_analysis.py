"""
Age Group Subgroup Analysis for NSCH 2023 Data
Analyzes screen time and mental health patterns across age groups
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
print("AGE GROUP SUBGROUP ANALYSIS (NSCH 2023)")
print("="*80)

# Load data
print("\n1. Loading NSCH data...")
df = pd.read_csv(DATA_PATH)
print(f"   Loaded: {len(df):,} children")

# Extract variables
print("\n2. Extracting and processing variables...")

df['age'] = pd.to_numeric(df['SC_AGE_YEARS'], errors='coerce')
df['screen_time'] = pd.to_numeric(df['SCREENTIME'], errors='coerce')
df['sleep_hours'] = pd.to_numeric(df['HOURSLEEP'], errors='coerce')

# Mental health indicators
df['depression_anxiety'] = df['K2Q32A'].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)
df['behavior_problem'] = df['K2Q33A'].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)
df['adhd'] = df['K2Q31A'].apply(lambda x: 1 if x == 1 else 0 if x == 2 else np.nan)
df['any_mental_health'] = ((df['depression_anxiety'] == 1) | 
                            (df['behavior_problem'] == 1) | 
                            (df['adhd'] == 1)).astype(int)

# Physical activity (1-7 scale typically)
df['physical_activity'] = pd.to_numeric(df['PHYSACTIV'], errors='coerce')

# Filter valid data
df_clean = df[
    (df['age'].notna()) & 
    (df['age'] >= 6) & 
    (df['age'] <= 17) &
    (df['screen_time'].notna()) &
    (df['screen_time'] <= 24)
].copy()

print(f"   Valid cases: {len(df_clean):,}")

# Create age groups
print("\n3. Creating age groups...")
df_clean['age_group'] = pd.cut(
    df_clean['age'],
    bins=[5, 9, 12, 14, 18],
    labels=['6-9 years', '10-12 years', '13-14 years', '15-17 years'],
    include_lowest=True
)

# Summary by age group
print("\n4. Computing summary statistics by age group...")
summary = df_clean.groupby('age_group').agg({
    'age': 'count',
    'screen_time': ['mean', 'std'],
    'sleep_hours': ['mean', 'std'],
    'depression_anxiety': 'mean',
    'behavior_problem': 'mean',
    'adhd': 'mean',
    'any_mental_health': 'mean'
}).round(3)

print(summary)

# VISUALIZATION 1: Screen Time Trends by Age Group
print("\n5. Generating Visualization 1: Screen time & sleep patterns...")

fig, axes = plt.subplots(2, 2, figsize=(32, 24))

# Plot 1: Screen time distribution by age group
age_groups = df_clean['age_group'].cat.categories
screen_data = [df_clean[df_clean['age_group'] == ag]['screen_time'].dropna() for ag in age_groups]

bp = axes[0, 0].boxplot(
    screen_data,
    labels=age_groups,
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='red', markersize=12)
)

for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[0, 0].set_xlabel('Age Group', fontsize=42, fontweight='bold')
axes[0, 0].set_ylabel('Daily Screen Time (hours)', fontsize=42, fontweight='bold')
axes[0, 0].set_title('Screen Time Distribution by Age Group', fontsize=44, fontweight='bold', pad=30)
axes[0, 0].tick_params(axis='both', labelsize=36)
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_xticklabels(age_groups, rotation=15, ha='right')

# Plot 2: Average screen time with error bars
screen_summary = df_clean.groupby('age_group')['screen_time'].agg(['mean', 'std', 'count']).reset_index()
x_pos = np.arange(len(screen_summary))

bars = axes[0, 1].bar(
    x_pos,
    screen_summary['mean'],
    yerr=screen_summary['std'] / np.sqrt(screen_summary['count']),
    color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
    edgecolor='black',
    linewidth=1.5,
    alpha=0.8,
    capsize=5
)

for bar, val, n in zip(bars, screen_summary['mean'], screen_summary['count']):
    height = bar.get_height()
    axes[0, 1].text(
        bar.get_x() + bar.get_width()/2,
        height + 0.3,
        f'{val:.1f}h\n(n={n:,})',
        ha='center',
        va='bottom',
        fontsize=16,
        fontweight='bold'
    )

axes[0, 1].set_xlabel('Age Group', fontsize=42, fontweight='bold')
axes[0, 1].set_ylabel('Average Daily Screen Time (hours)', fontsize=42, fontweight='bold')
axes[0, 1].set_title('Mean Screen Time by Age Group', fontsize=44, fontweight='bold', pad=30)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(screen_summary['age_group'], rotation=15, ha='right', fontsize=36)
axes[0, 1].tick_params(axis='y', labelsize=36)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Sleep hours by age group
sleep_summary = df_clean.groupby('age_group')['sleep_hours'].agg(['mean', 'std', 'count']).reset_index()

bars = axes[1, 0].bar(
    x_pos,
    sleep_summary['mean'],
    yerr=sleep_summary['std'] / np.sqrt(sleep_summary['count']),
    color=['#9b59b6', '#34495e', '#16a085', '#d35400'],
    edgecolor='black',
    linewidth=1.5,
    alpha=0.8,
    capsize=5
)

for bar, val in zip(bars, sleep_summary['mean']):
    height = bar.get_height()
    axes[1, 0].text(
        bar.get_x() + bar.get_width()/2,
        height + 0.25,
        f'{val:.1f}h',
        ha='center',
        va='bottom',
        fontsize=18,
        fontweight='bold'
    )

axes[1, 0].set_xlabel('Age Group', fontsize=42, fontweight='bold')
axes[1, 0].set_ylabel('Average Sleep Hours', fontsize=42, fontweight='bold')
axes[1, 0].set_title('Sleep Duration by Age Group', fontsize=44, fontweight='bold', pad=30)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(sleep_summary['age_group'], rotation=15, ha='right', fontsize=36)
axes[1, 0].tick_params(axis='y', labelsize=36)
axes[1, 0].axhline(y=9, color='red', linestyle='--', linewidth=2, label='Recommended (9h)')
axes[1, 0].legend(fontsize=18)
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Screen time vs sleep scatter by age group
for ag, color in zip(age_groups, ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']):
    ag_data = df_clean[df_clean['age_group'] == ag][['screen_time', 'sleep_hours']].dropna()
    if len(ag_data) > 100:
        sample = ag_data.sample(min(500, len(ag_data)))
        axes[1, 1].scatter(
            sample['screen_time'],
            sample['sleep_hours'],
            alpha=0.3,
            s=20,
            color=color,
            label=ag
        )

axes[1, 1].set_xlabel('Daily Screen Time (hours)', fontsize=42, fontweight='bold')
axes[1, 1].set_ylabel('Sleep Hours', fontsize=42, fontweight='bold')
axes[1, 1].set_title('Screen Time vs Sleep Duration', fontsize=44, fontweight='bold', pad=30)
axes[1, 1].tick_params(axis='both', labelsize=36)
axes[1, 1].legend(fontsize=32, loc='upper right')
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Screen Time & Sleep Patterns by Age Group (NSCH 2023)', 
             fontsize=48, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f"{OUTPUT_DIR}/age_group_screen_sleep_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/age_group_screen_sleep_analysis.png")

# VISUALIZATION 2: Mental Health by Age Group
print("\n6. Generating Visualization 2: Mental health prevalence...")

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
    
    prevalence = df_clean.groupby('age_group')[col].agg(['mean', 'count']).reset_index()
    prevalence['prevalence_pct'] = prevalence['mean'] * 100
    
    bars = ax.bar(
        range(len(prevalence)),
        prevalence['prevalence_pct'],
        color=color,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    for bar, val, n in zip(bars, prevalence['prevalence_pct'], prevalence['count']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 1.5,
            f'{val:.1f}%\n(n={n:,})',
            ha='center',
            va='bottom',
            fontsize=18,
            fontweight='bold'
        )
    
    ax.set_xlabel('Age Group', fontsize=42, fontweight='bold')
    ax.set_ylabel('Prevalence (%)', fontsize=42, fontweight='bold')
    ax.set_title(f'{title}', fontsize=44, fontweight='bold', pad=30)
    ax.set_xticks(range(len(prevalence)))
    ax.set_xticklabels(prevalence['age_group'], rotation=15, ha='right', fontsize=36)
    ax.tick_params(axis='y', labelsize=36)
    ax.set_ylim([0, max(prevalence['prevalence_pct']) * 1.3])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Mental Health Prevalence by Age Group (NSCH 2023)', 
             fontsize=48, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(f"{OUTPUT_DIR}/age_group_mental_health_prevalence.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/age_group_mental_health_prevalence.png")

# VISUALIZATION 3: Comprehensive Age Group Comparison Table
print("\n7. Generating Visualization 3: Comprehensive comparison table...")

fig, ax = plt.subplots(figsize=(40, 26))
ax.axis('tight')
ax.axis('off')

# Prepare comprehensive data
table_data = [['Age Group', 'N', 'Screen Time\n(hours/day)', 'Sleep\n(hours)', 
               'Depression/\nAnxiety (%)', 'Behavioral\nProblems (%)', 
               'ADHD (%)', 'Any Mental\nHealth (%)']]

for ag in age_groups:
    ag_data = df_clean[df_clean['age_group'] == ag]
    row = [
        str(ag),
        f"{len(ag_data):,}",
        f"{ag_data['screen_time'].mean():.1f} ± {ag_data['screen_time'].std():.1f}",
        f"{ag_data['sleep_hours'].mean():.1f} ± {ag_data['sleep_hours'].std():.1f}",
        f"{ag_data['depression_anxiety'].mean() * 100:.1f}%",
        f"{ag_data['behavior_problem'].mean() * 100:.1f}%",
        f"{ag_data['adhd'].mean() * 100:.1f}%",
        f"{ag_data['any_mental_health'].mean() * 100:.1f}%"
    ]
    table_data.append(row)

table = ax.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    colWidths=[0.15, 0.10, 0.15, 0.12, 0.12, 0.12, 0.10, 0.14]
)

table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 3.5)

# Style header row
for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontsize=15)

# Style data rows
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if j == 0:
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(weight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#ffffff')
        else:
            cell.set_facecolor('#f8f9fa')

ax.set_title('Comprehensive Age Group Comparison: Screen Time, Sleep & Mental Health (NSCH 2023)', 
             fontsize=28, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/age_group_comprehensive_table.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   [OK] Saved: {OUTPUT_DIR}/age_group_comprehensive_table.png")

# Summary output
print("\n" + "="*80)
print("AGE GROUP ANALYSIS COMPLETE - KEY FINDINGS:")
print("="*80)

for ag in age_groups:
    ag_data = df_clean[df_clean['age_group'] == ag]
    print(f"\n{ag} (n={len(ag_data):,}):")
    print(f"  - Screen Time: {ag_data['screen_time'].mean():.1f} ± {ag_data['screen_time'].std():.1f} hours/day")
    print(f"  - Sleep: {ag_data['sleep_hours'].mean():.1f} ± {ag_data['sleep_hours'].std():.1f} hours")
    print(f"  - Depression/Anxiety: {ag_data['depression_anxiety'].mean() * 100:.1f}%")
    print(f"  - Behavioral Problems: {ag_data['behavior_problem'].mean() * 100:.1f}%")
    print(f"  - ADHD: {ag_data['adhd'].mean() * 100:.1f}%")
    print(f"  - Any Mental Health: {ag_data['any_mental_health'].mean() * 100:.1f}%")

print("\n" + "="*80)
print("Generated 3 age group subgroup visualizations:")
print("  1. age_group_screen_sleep_analysis.png")
print("  2. age_group_mental_health_prevalence.png")
print("  3. age_group_comprehensive_table.png")
print("="*80)
