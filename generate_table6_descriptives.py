"""
Generate Table 6: Descriptive Statistics of Dataset
Based on current cleaned data (310 samples)
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_excel('data/IGD Database.xlsx')

# Convert categorical variables to numeric
def convert_hours_to_numeric(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    mapping = {
        '1 or less': 0.5, '1 to 2': 1.5, '2 to 3': 2.5, '3 to 4': 3.5, '4 to 5': 4.5,
        '5 to 6': 5.5, '6 to 7': 6.5, '7 to 8': 7.5, '8 to 9': 8.5, '9 to 10': 9.5,
        '10 or more': 10.5,
    }
    return mapping.get(val, np.nan)

df['Weekday Hours'] = df['Weekday Hours'].apply(convert_hours_to_numeric)
df['Weekend Hours'] = df['Weekend Hours'].apply(convert_hours_to_numeric)

# Convert other features to numeric
for col in ['Sleep Quality', 'IGD Total', 'Social', 'Escape']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Select features and target
features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 'IGD Total', 'Social', 'Escape']
target = 'IGD Status'

# Drop rows with any missing values in features (SAME AS MODEL PREPROCESSING)
X = df[features].dropna()

# Convert target to binary
if df[target].dtype == 'object':
    y = (df.loc[X.index, target].astype(str).str.strip() == 'Y').astype(int)
else:
    y = df.loc[X.index, target].astype(int)

# Remove any NaN in target
mask = ~y.isna()
X = X[mask]
y = y[mask]

# Get corresponding demographic data
df_clean = df.loc[X.index].copy()

print("="*80)
print("TABLE 6: DESCRIPTIVE STATISTICS OF DATASET")
print("(After removing missing values - Final analytical sample)")
print("="*80)

# Create results table
results = []

# Demographics header
results.append(["Demographics", ""])
results.append(["Age (years)", f"{df_clean['Age'].mean():.1f} ± {df_clean['Age'].std():.1f}"])

# Gender counts - column is 'Sex' not 'Gender'
if df_clean['Sex'].dtype == 'object':
    male_count = df_clean['Sex'].str.strip().isin(['M', 'Male']).sum()
    female_count = df_clean['Sex'].str.strip().isin(['F', 'Female']).sum()
else:
    male_count = (df_clean['Sex'] == 1).sum()
    female_count = (df_clean['Sex'] == 0).sum()

total_n = len(df_clean)
results.append(["Male", f"{male_count} ({male_count/total_n*100:.1f}%)"])
results.append(["Female", f"{female_count} ({female_count/total_n*100:.1f}%)"])

# Gaming Behaviour
results.append(["Gaming Behaviour", ""])
results.append(["Weekday gaming (hours/day)", f"{X['Weekday Hours'].mean():.1f} ± {X['Weekday Hours'].std():.1f}"])
results.append(["Weekend gaming (hours/day)", f"{X['Weekend Hours'].mean():.1f} ± {X['Weekend Hours'].std():.1f}"])

# Sleep
results.append(["Sleep", ""])
results.append(["Sleep quality (1-5)", f"{X['Sleep Quality'].mean():.1f} ± {X['Sleep Quality'].std():.1f}"])

# IGD Measures
results.append(["IGD Measures", ""])
results.append(["IGD total score", f"{X['IGD Total'].mean():.1f} ± {X['IGD Total'].std():.1f}"])
results.append(["IGD-positive cases", f"{y.sum()} ({y.sum()/len(y)*100:.1f}%)"])

# Motivations (MOGQ)
results.append(["Motivations (MOGQ)", ""])
results.append(["Social motivation", f"{X['Social'].mean():.1f} ± {X['Social'].std():.1f}"])
results.append(["Escape motivation", f"{X['Escape'].mean():.1f} ± {X['Escape'].std():.1f}"])

# Print table
print(f"\n{'Variable':<40} {'Mean ± SD / N (%)':<30}")
print("="*80)
for row in results:
    if row[1] == "":  # Header rows
        print(f"\n{row[0]}")
    else:
        print(f"{row[0]:<40} {row[1]:<30}")

print("\n" + "="*80)
print(f"Total Sample Size: {len(df_clean)}")
print(f"Original Dataset: 395 participants")
print(f"Cases Removed (missing data): {395 - len(df_clean)} ({(395-len(df_clean))/395*100:.1f}%)")
print("="*80)

# Save to CSV for easy copying
table_df = pd.DataFrame(results, columns=['Variable', 'Mean ± SD / N (%)'])
table_df.to_csv('visualizations/table6_descriptive_statistics.csv', index=False)
print("\n✓ Table saved to: visualizations/table6_descriptive_statistics.csv")
