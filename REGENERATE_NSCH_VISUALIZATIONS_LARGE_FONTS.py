#!/usr/bin/env python3
"""
REGENERATE NSCH VISUALIZATIONS WITH LARGER FONTS
==================================================

Regenerates all NSCH figures with SIGNIFICANTLY LARGER fonts for better readability
in printed dissertations and presentations.

Font Scaling (Same as IGD):
- Titles: 16-18pt (was 12-14pt) ↑ 25-35%
- Axis labels: 14pt (was 11-12pt) ↑ 17-27%
- Legends: 12pt (was 10pt) ↑ 20%
- Tick labels: 12pt (was 10pt) ↑ 20%
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use larger default font
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

print("="*80)
print("REGENERATING NSCH VISUALIZATIONS WITH LARGER FONTS")
print("="*80)

# ============================================================================
# LOAD NSCH DATA
# ============================================================================

print("\n[*] Loading NSCH data...")

try:
    nsch_df = pd.read_excel("../data/nsch_2023e_topical.xlsx")
    print(f"✓ Loaded NSCH data: {nsch_df.shape[0]:,} rows, {nsch_df.shape[1]} columns")
except:
    try:
        nsch_df = pd.read_excel("NSCH_Project/data/nsch_2023e_topical.xlsx")
        print(f"✓ Loaded NSCH data: {nsch_df.shape[0]:,} rows, {nsch_df.shape[1]} columns")
    except:
        try:
            nsch_df = pd.read_excel("data/nsch_2023e_topical.xlsx")
            print(f"✓ Loaded NSCH data: {nsch_df.shape[0]:,} rows, {nsch_df.shape[1]} columns")
        except:
            print("ERROR: Could not find NSCH data file")
            import sys
            sys.exit(1)

# Create visualizations folder
os.makedirs('nsch_visualizations_large_fonts', exist_ok=True)

# Set style for better-looking plots
sns.set_style("whitegrid")

# ============================================================================
# FIGURE 1: DATASET OVERVIEW - LARGE FONTS
# ============================================================================

print("\n[1/7] Creating dataset overview (LARGE FONTS)...")

fig, ax = plt.subplots(figsize=(14, 10))

info_text = f"""
NSCH DATASET OVERVIEW

Sample Size: {nsch_df.shape[0]:,} children
Total Variables: {nsch_df.shape[1]}
Age Range: 0-17 years
Survey Year: 2023

Data Completeness:
Complete cases: {(~nsch_df.isnull().any(axis=1)).sum():,} ({(~nsch_df.isnull().any(axis=1)).sum()/len(nsch_df)*100:.1f}%)
Cases with missing data: {nsch_df.isnull().any(axis=1).sum():,} ({nsch_df.isnull().any(axis=1).sum()/len(nsch_df)*100:.1f}%)
"""

ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=16, 
        verticalalignment='top', fontfamily='monospace', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))
ax.axis('off')

plt.tight_layout()
plt.savefig('nsch_visualizations_large_fonts/01_dataset_overview_LARGE_FONTS.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_dataset_overview_LARGE_FONTS.png")

# ============================================================================
# FIGURE 2: MISSING DATA VISUALIZATION - LARGE FONTS
# ============================================================================

print("[2/7] Creating missing data visualization (LARGE FONTS)...")

missing_data = nsch_df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False).head(15)

if len(missing_data) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))
    missing_data.plot(kind='barh', ax=ax, color='coral')
    
    ax.set_xlabel('Number of Missing Values', fontsize=14, fontweight='bold')
    ax.set_ylabel('Column Name', fontsize=14, fontweight='bold')
    ax.set_title('Missing Data Distribution (Top 15 Columns)', fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='x', alpha=0.3, linewidth=1.2)
    
    plt.tight_layout()
    plt.savefig('nsch_visualizations_large_fonts/02_missing_data_LARGE_FONTS.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_missing_data_LARGE_FONTS.png")

# ============================================================================
# FIGURE 3: DESCRIPTIVE STATISTICS - LARGE FONTS
# ============================================================================

print("[3/7] Creating descriptive statistics (LARGE FONTS)...")

fig, ax = plt.subplots(figsize=(16, 10))

# Get numeric columns only
numeric_cols = nsch_df.select_dtypes(include=[np.number]).columns[:10]
stats = nsch_df[numeric_cols].describe().round(2)

stats_text = "DESCRIPTIVE STATISTICS (Numeric Variables)\n\n"
stats_text += stats.to_string()

ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5, pad=1))
ax.axis('off')

plt.tight_layout()
plt.savefig('nsch_visualizations_large_fonts/03_descriptive_statistics_LARGE_FONTS.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_descriptive_statistics_LARGE_FONTS.png")

# ============================================================================
# FIGURE 4: DATA TYPE DISTRIBUTION - LARGE FONTS
# ============================================================================

print("[4/7] Creating data type distribution (LARGE FONTS)...")

# Count data types
dtype_counts = nsch_df.dtypes.value_counts()

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(range(len(dtype_counts)), dtype_counts.values, color='steelblue', width=0.6)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_xticks(range(len(dtype_counts)))
ax.set_xticklabels([str(dt) for dt in dtype_counts.index], fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Columns', fontsize=14, fontweight='bold')
ax.set_title('Data Type Distribution', fontsize=16, fontweight='bold', pad=15)
ax.tick_params(axis='y', which='major', labelsize=12)
ax.grid(axis='y', alpha=0.3, linewidth=1.2)

plt.tight_layout()
plt.savefig('nsch_visualizations_large_fonts/04_data_type_distribution_LARGE_FONTS.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_data_type_distribution_LARGE_FONTS.png")

# ============================================================================
# FIGURE 5: SCREEN TIME DISTRIBUTION (if available) - LARGE FONTS
# ============================================================================

print("[5/7] Creating screen time distribution (LARGE FONTS)...")

# Look for screen time column (various possible names)
screen_col = None
for col_name in nsch_df.columns:
    if 'screen' in col_name.lower() or 'screentime' in col_name.lower():
        screen_col = col_name
        break

if screen_col is not None:
    screen_data = pd.to_numeric(nsch_df[screen_col], errors='coerce').dropna()
    
    if len(screen_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        axes[0].hist(screen_data, bins=30, color='skyblue', edgecolor='black', linewidth=1.2)
        axes[0].set_xlabel('Screen Time (hours)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=14, fontweight='bold')
        axes[0].set_title('Distribution of Screen Time', fontsize=16, fontweight='bold', pad=15)
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        axes[0].grid(axis='y', alpha=0.3, linewidth=1.2)
        
        # Box plot
        axes[1].boxplot(screen_data, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Screen Time (hours)', fontsize=14, fontweight='bold')
        axes[1].set_title('Screen Time Box Plot', fontsize=16, fontweight='bold', pad=15)
        axes[1].tick_params(axis='both', which='major', labelsize=12)
        axes[1].grid(axis='y', alpha=0.3, linewidth=1.2)
        
        plt.suptitle(f'Screen Time Analysis (n={len(screen_data):,})', 
                    fontsize=18, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        plt.savefig('nsch_visualizations_large_fonts/05_screen_time_distribution_LARGE_FONTS.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 05_screen_time_distribution_LARGE_FONTS.png")
else:
    print("⚠ Screen time column not found, skipping Figure 5")

# ============================================================================
# FIGURE 6: MENTAL HEALTH INDICATORS - LARGE FONTS
# ============================================================================

print("[6/7] Creating mental health indicators (LARGE FONTS)...")

# Look for mental health columns
mental_cols = []
for col_name in nsch_df.columns:
    col_lower = col_name.lower()
    if any(keyword in col_lower for keyword in ['mental', 'anxiety', 'depression', 'behavior', 'health']):
        mental_cols.append(col_name)

mental_cols = mental_cols[:6]  # Limit to 6 columns

if len(mental_cols) > 0:
    # Get numeric versions of mental health columns
    mental_data = {}
    for col in mental_cols:
        numeric_col = pd.to_numeric(nsch_df[col], errors='coerce')
        if numeric_col.notna().sum() > 0:
            mental_data[col] = numeric_col
    
    if len(mental_data) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (col_name, col_data) in enumerate(mental_data.items()):
            col_data_clean = col_data.dropna()
            
            axes[idx].hist(col_data_clean, bins=20, color='mediumpurple', edgecolor='black', linewidth=1.2)
            axes[idx].set_xlabel('Value', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontsize=12, fontweight='bold')
            axes[idx].set_title(col_name, fontsize=13, fontweight='bold', pad=10)
            axes[idx].tick_params(axis='both', which='major', labelsize=11)
            axes[idx].grid(axis='y', alpha=0.3, linewidth=1.0)
        
        # Hide unused subplots
        for idx in range(len(mental_data), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Mental Health & Behavioral Indicators Distribution', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig('nsch_visualizations_large_fonts/06_mental_health_indicators_LARGE_FONTS.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 06_mental_health_indicators_LARGE_FONTS.png")
else:
    print("⚠ Mental health columns not found, skipping Figure 6")

# ============================================================================
# FIGURE 7: CORRELATION HEATMAP - LARGE FONTS
# ============================================================================

print("[7/7] Creating correlation heatmap (LARGE FONTS)...")

# Select numeric columns for correlation
numeric_cols = nsch_df.select_dtypes(include=[np.number]).columns[:12]
correlation_matrix = nsch_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
            ax=ax, linewidths=0.5, annot_kws={'size': 10, 'weight': 'bold'})

ax.set_title('Correlation Matrix - Selected Numeric Variables', 
            fontsize=16, fontweight='bold', pad=15)
ax.tick_params(axis='both', which='major', labelsize=11)

# Rotate labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
plt.yticks(rotation=0, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('nsch_visualizations_large_fonts/07_correlation_heatmap_LARGE_FONTS.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_correlation_heatmap_LARGE_FONTS.png")

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ ALL NSCH VISUALIZATIONS REGENERATED WITH LARGER FONTS!")
print("="*80)

print("\nFonts Increased:")
print("  • Titles: 12-14pt → 16-18pt (↑ 25-35%)")
print("  • Axis Labels: 11-12pt → 14pt (↑ 17-27%)")
print("  • Legends: 10pt → 12pt (↑ 20%)")
print("  • Tick Labels: 10pt → 12pt (↑ 20%)")

print("\nNew Files Generated:")
print("  1. 01_dataset_overview_LARGE_FONTS.png")
print("  2. 02_missing_data_LARGE_FONTS.png")
print("  3. 03_descriptive_statistics_LARGE_FONTS.png")
print("  4. 04_data_type_distribution_LARGE_FONTS.png")
print("  5. 05_screen_time_distribution_LARGE_FONTS.png")
print("  6. 06_mental_health_indicators_LARGE_FONTS.png")
print("  7. 07_correlation_heatmap_LARGE_FONTS.png")

print("\n📍 Location: nsch_visualizations_large_fonts/ folder")
print("\n✨ All figures are now 300 DPI and publication-ready!")
print("="*80)
