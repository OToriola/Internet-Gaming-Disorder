"""
NSCH Descriptive Data Visualization Script (Screenshot-Ready)
Generates basic exploratory visualizations to show what the NSCH data looks like
- BIG fonts (titles/labels/ticks/text boxes) for easy screenshots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIG: PATHS + GLOBAL STYLING (BIG FONTS)
# =============================================================================
DATA_PATH = "data/nsch_2023e_topical.xlsx"
OUT_DIR = "visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

# Seaborn style
sns.set_style("whitegrid")

# BIG global font sizes for screenshots
plt.rcParams.update({
    "figure.figsize": (16, 9),
    "figure.dpi": 120,
    "savefig.dpi": 300,

    "font.size": 24,
    "axes.titlesize": 40,
    "axes.labelsize": 34,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 26,
    "figure.titlesize": 44,

    "axes.titlepad": 20,
    "axes.labelpad": 14,
})

def save_fig(path: str):
    """Consistent save settings."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def style_axes(ax):
    """Make axes/ticks thick and readable for screenshots."""
    ax.tick_params(axis="both", which="major", length=8, width=2, labelsize=28)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

print("Loading NSCH data...")
df = pd.read_excel(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"Total children: {df.shape[0]:,}")
print(f"Total variables: {df.shape[1]}")

# =============================================================================
# 1. BASIC DATA OVERVIEW (TEXT CARD)
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING BASIC DESCRIPTIVE VISUALIZATIONS (BIG FONT MODE)")
print("=" * 70)

missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df) * 100).round(2)

top_missing = (
    pd.DataFrame({"missing_n": missing_counts, "missing_%": missing_pct})
      .query("missing_n > 0")
      .sort_values("missing_n", ascending=False)
      .head(15)
)

overview_lines = [
    "NSCH DATASET OVERVIEW (2023)",
    "",
    f"Total Sample Size: {df.shape[0]:,} children",
    f"Total Variables:   {df.shape[1]}",
    "Age Range:         0-17 years",
    "Survey Year:       2023",
    "",
    "Top Missingness (Top 15 columns):"
]

if len(top_missing) == 0:
    overview_lines.append("  • No missing values detected.")
else:
    for col, row in top_missing.iterrows():
        overview_lines.append(f"  • {col}: {int(row['missing_n']):,} ({row['missing_%']:.2f}%)")

info_text = "\n".join(overview_lines)

fig, ax = plt.subplots(figsize=(18, 10))
ax.axis("off")
ax.text(
    0.03, 0.97, info_text,
    transform=ax.transAxes,
    fontsize=28,  # BIG
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=1.0", facecolor="wheat", alpha=0.5)
)
save_fig(os.path.join(OUT_DIR, "01_dataset_overview.png"))
print("✓ Saved: 01_dataset_overview.png")

# =============================================================================
# 2. MISSING DATA VISUALIZATION
# =============================================================================
missing_data = missing_counts[missing_counts > 0].sort_values(ascending=False).head(15)

if len(missing_data) > 0:
    fig, ax = plt.subplots(figsize=(18, 10))
    missing_data.sort_values().plot(kind="barh", ax=ax, color="coral", edgecolor="black")

    ax.set_title("Missing Data by Column (Top 15)", fontsize=42, fontweight="bold")
    ax.set_xlabel("Number of Missing Values", fontsize=36)
    ax.set_ylabel("Column", fontsize=36)

    style_axes(ax)
    save_fig(os.path.join(OUT_DIR, "02_missing_data.png"))
    print("✓ Saved: 02_missing_data.png")
else:
    print("✓ Skipped: 02_missing_data.png (no missing values)")

# =============================================================================
# 3. BASIC STATISTICS TABLE (TEXT CARD)
# =============================================================================
print("\nGenerating basic statistics...")
numeric_df_all = df.select_dtypes(include=[np.number])
if numeric_df_all.shape[1] > 0:
    stats = df.describe().T
    print("\nBasic Statistics (Numeric Columns):")
    print(stats.head(10))

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis("off")

    stats_display = df.describe().round(2).to_string()
    ax.text(
        0.02, 0.98,
        "DESCRIPTIVE STATISTICS\n\n" + stats_display,
        transform=ax.transAxes,
        fontsize=20,  # still large; table is long
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="lightblue", alpha=0.3)
    )
    save_fig(os.path.join(OUT_DIR, "03_descriptive_statistics.png"))
    print("✓ Saved: 03_descriptive_statistics.png")
else:
    print("✓ Skipped: 03_descriptive_statistics.png (no numeric columns)")

# =============================================================================
# 4. DATA TYPE DISTRIBUTION
# =============================================================================
dtype_counts = df.dtypes.value_counts()

fig, ax = plt.subplots(figsize=(14, 9))
dtype_counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")

ax.set_title("Data Types in NSCH Dataset", fontsize=42, fontweight="bold")
ax.set_xlabel("Data Type", fontsize=36)
ax.set_ylabel("Count", fontsize=36)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

style_axes(ax)
save_fig(os.path.join(OUT_DIR, "04_data_types.png"))
print("✓ Saved: 04_data_types.png")

# =============================================================================
# 5. NUMERIC COLUMNS DISTRIBUTION (HISTOGRAMS)
# =============================================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nFound {len(numeric_cols)} numeric columns")

if len(numeric_cols) > 0:
    sample_cols = numeric_cols[:6]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, col in enumerate(sample_cols):
        ax = axes[idx]
        ax.hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="black", alpha=0.75)
        ax.set_title(str(col), fontsize=32, fontweight="bold")
        ax.set_xlabel("Value", fontsize=28)
        ax.set_ylabel("Frequency", fontsize=28)
        style_axes(ax)

    # If fewer than 6 columns, hide unused subplots
    for j in range(len(sample_cols), 6):
        axes[j].axis("off")

    fig.suptitle("Distribution of Numeric Variables (Sample)", fontsize=46, fontweight="bold", y=1.02)
    save_fig(os.path.join(OUT_DIR, "05_numeric_distributions.png"))
    print("✓ Saved: 05_numeric_distributions.png")
else:
    print("✓ Skipped: 05_numeric_distributions.png (no numeric columns)")

# =============================================================================
# 6. CATEGORICAL COLUMNS OVERVIEW (TOP 10 COUNTS)
# =============================================================================
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"Found {len(categorical_cols)} categorical columns")

if len(categorical_cols) > 0:
    sample_cols = categorical_cols[:6]

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    axes = axes.flatten()

    for idx, col in enumerate(sample_cols):
        ax = axes[idx]
        value_counts = df[col].value_counts(dropna=False).head(10)

        ax.barh(range(len(value_counts)), value_counts.values, color="coral", edgecolor="black")
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels([str(x) for x in value_counts.index], fontsize=22)
        ax.invert_yaxis()

        ax.set_title(f"{col} (Top 10)", fontsize=32, fontweight="bold")
        ax.set_xlabel("Count", fontsize=28)
        style_axes(ax)

    for j in range(len(sample_cols), 6):
        axes[j].axis("off")

    fig.suptitle("Distribution of Categorical Variables (Sample)", fontsize=46, fontweight="bold", y=1.02)
    save_fig(os.path.join(OUT_DIR, "06_categorical_distributions.png"))
    print("✓ Saved: 06_categorical_distributions.png")
else:
    print("✓ Skipped: 06_categorical_distributions.png (no object columns)")

# =============================================================================
# 7. CORRELATION MATRIX (NUMERIC ONLY)
# =============================================================================
print("\nGenerating correlation matrix...")
numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.shape[1] > 1:
    corr_cols = numeric_df.columns[:20] if numeric_df.shape[1] > 20 else numeric_df.columns
    corr_matrix = numeric_df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title("Correlation Matrix - Numeric Variables (Sample)", fontsize=42, fontweight="bold", pad=20)
    ax.tick_params(axis="both", labelsize=22)  # heatmap labels can get crowded
    save_fig(os.path.join(OUT_DIR, "07_correlation_matrix.png"))
    print("✓ Saved: 07_correlation_matrix.png")
else:
    print("✓ Skipped: 07_correlation_matrix.png (not enough numeric columns)")

# =============================================================================
# 8. DATA QUALITY SUMMARY (TEXT CARD)
# =============================================================================
print("\nGenerating data quality report...")

quality_metrics = {
    "Total Rows": df.shape[0],
    "Total Columns": df.shape[1],
    "Complete Cases": int((~df.isnull().any(axis=1)).sum()),
    "Rows with Missing": int(df.isnull().any(axis=1).sum()),
    "Total Missing Values": int(df.isnull().sum().sum()),
    "Completeness %": round((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
}

quality_text = "DATA QUALITY REPORT\n\n"
for key, value in quality_metrics.items():
    quality_text += f"{key:<22}: {value}\n"

fig, ax = plt.subplots(figsize=(16, 9))
ax.axis("off")
ax.text(
    0.05, 0.95, quality_text,
    transform=ax.transAxes,
    fontsize=30,  # BIG
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=1.0", facecolor="lightgreen", alpha=0.3)
)
save_fig(os.path.join(OUT_DIR, "08_data_quality_report.png"))
print("✓ Saved: 08_data_quality_report.png")

# =============================================================================
# 9. DATASET SUMMARY (TEXT CARD)
# =============================================================================
print("\nGenerating sample summaries...")

summary_report = f"""
NSCH 2023 TOPICAL - DATASET SUMMARY

Sample Composition:
  • Total Children: {df.shape[0]:,}
  • Age Range: 0-17 years
  • Nationally Representative: Yes

Variables:
  • Numeric Variables: {len(numeric_cols)}
  • Categorical Variables: {len(categorical_cols)}
  • Total Variables: {df.shape[1]}

Data Quality:
  • Complete Cases: {quality_metrics['Complete Cases']:,}
  • Cases with Missing Data: {quality_metrics['Rows with Missing']:,}
  • Overall Completeness: {quality_metrics['Completeness %']:.2f}%

Key Features (typical in NSCH topical extracts):
  • Screen Time Data
  • Mental Health Indicators
  • Sleep Information
  • Physical Activity Data
  • Demographic Information
  • Family Background
""".strip("\n")

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis("off")
ax.text(
    0.05, 0.95, summary_report,
    transform=ax.transAxes,
    fontsize=28,  # BIG
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=1.0", facecolor="lightyellow", alpha=0.6)
)
save_fig(os.path.join(OUT_DIR, "09_dataset_summary.png"))
print("✓ Saved: 09_dataset_summary.png")

# =============================================================================
# COMPLETION MESSAGE
# =============================================================================
print("\n" + "=" * 70)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated Files:")
print("  1. 01_dataset_overview.png - Basic dataset information (big text)")
print("  2. 02_missing_data.png - Missing data by column (big axes)")
print("  3. 03_descriptive_statistics.png - Statistical summary (text card)")
print("  4. 04_data_types.png - Distribution of data types (big axes)")
print("  5. 05_numeric_distributions.png - Histograms of numeric variables")
print("  6. 06_categorical_distributions.png - Bar charts of categorical variables")
print("  7. 07_correlation_matrix.png - Correlation heatmap")
print("  8. 08_data_quality_report.png - Data quality metrics (big text)")
print("  9. 09_dataset_summary.png - Complete dataset summary (big text)")
print(f"\nAll saved to: {OUT_DIR}/")
print("=" * 70)
