# Ensure data is loaded
import pandas as pd
df = pd.read_csv('nsch2023_merged_reduced.csv')

# --- Clear ADHD Prevalence Bar Chart: Ever, Current, Never ---
ever = (df['K2Q31A'] == 1).sum()
current = (df['K2Q31B'] == 1).sum()
never = (df['K2Q31A'] == 2).sum()

labels = ['Ever Diagnosed', 'Currently Diagnosed', 'Never Diagnosed']
counts = [ever, current, never]

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,5))
sns.barplot(x=labels, y=counts, palette='Set2')
plt.title('ADHD Prevalence: Ever, Current, and Never Diagnosed')
plt.ylabel('Number of Children')
plt.xlabel('Diagnosis Status')
plt.tight_layout()
plt.show()

print(f"Ever diagnosed: {ever}")
print(f"Currently diagnosed: {current}")
print(f"Never diagnosed: {never}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# --- Load variable dictionary for user-friendly labels ---
import csv
os.makedirs('plots', exist_ok=True)

# Build a dictionary: variable -> (description, question)
var_dict = {}
with open('NSCH_Dictionary_22-08-2025_0315_pm(NSCH_Dictionary_22-08-2025_0315).csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        var = row['Variable']
        desc = row['Description']
        q = row['Question']
        var_dict[var] = (desc, q)

# Load data
df = pd.read_csv('nsch2023_merged_reduced.csv')

# 1. Data Overview
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Missing values per column:')
print(df.isnull().sum())
# Interpretation:
# This gives a sense of dataset size, available variables, and where missing data may affect analysis.

# 2. Univariate Analysis
cat_vars = ['K2Q31A', 'K2Q31B', 'SCREENTIME', 'SC_SEX', 'c_sex', 'c_race_r']



import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Category mappings for clarity
category_maps = {
    'SC_SEX': {1: 'Male', 2: 'Female'},
    'K2Q31A': {1: 'Yes', 2: 'No'},
    'K2Q31B': {1: 'Yes', 2: 'No'},
    'SCREENTIME': {
        1: '<1 hour',
        2: '1 hour',
        3: '2 hours',
        4: '3 hours',
        5: '4+ hours'
    }
}

for var in cat_vars:
    if var in df.columns:
        desc, q = var_dict.get(var, (var, var))
        # Map categories if available
        if var in category_maps:
            df[var + '_label'] = df[var].map(category_maps[var])
            plot_x = var + '_label'
            xlabel = f"{desc} (" + ', '.join([f"{k}={v}" for k,v in category_maps[var].items()]) + ")"
        else:
            plot_x = var
            xlabel = desc
        print(f'\n{desc} ({var}): {q}')
        print('Value counts:')
        print(df[var].value_counts(dropna=False))
        plt.figure()
        sns.countplot(x=plot_x, data=df, palette='Set2')
        plt.title(f'{desc}')
        plt.xlabel(xlabel)
        plt.ylabel('Number of Children')
        plt.tight_layout()
        plt.savefig(f'plots/univariate_{var}.png')
        plt.close()
        # Plain-language summary
        print(f'Visual: See plots/univariate_{var}.png. This bar plot shows the distribution of responses for "{desc}". {q}')
        if var in category_maps:
            print(f'Category guide: ' + ', '.join([f"{k} = {v}" for k,v in category_maps[var].items()]))
        print('Interpretation: Look for which categories are most/least common. Imbalances may affect analysis.')

num_vars = ['SC_AGE_YEARS', 'c_age_years', 'SCREENTIME', 'HOURSLEEP', 'HOURSLEEP05']



for var in num_vars:
    if var in df.columns:
        # Manual override for HOURSLEEP and HOURSLEEP05
        if var == 'HOURSLEEP':
            desc = 'Sleep Hours (Weeknights)'
            q = 'During the past week, how many hours of sleep did this child get on most weeknights?'
        elif var == 'HOURSLEEP05':
            desc = 'Sleep Hours (All days)'
            q = 'During the past week, how many hours of sleep did this child get during an average day (count both nighttime sleep and naps)?'
        else:
            desc, q = var_dict.get(var, (var, var))
        print(f'\n{desc} ({var}): {q}')
        print('Summary stats:')
        print(df[var].describe())
        plt.figure()
        sns.histplot(df[var].dropna(), kde=True)
        plt.title(f'{desc}')
        plt.xlabel(desc)
        plt.ylabel('Number of Children')
        plt.tight_layout()
        plt.savefig(f'plots/univariate_{var}.png')
        plt.close()
        # Plain-language summary
        print(f'Visual: See plots/univariate_{var}.png. This histogram shows the distribution of "{desc}". {q}')
        print('Interpretation: Typical values, spread, and outliers can be seen. Compare to expected/healthy ranges.')

# 3. Bivariate Analysis


if 'K2Q31B' in df.columns and 'SCREENTIME' in df.columns:
    adhd_desc, adhd_q = var_dict.get('K2Q31B', ('K2Q31B', ''))
    st_desc, st_q = var_dict.get('SCREENTIME', ('SCREENTIME', ''))
    # Use mapped labels for clarity
    if 'K2Q31B_label' in df.columns and 'SCREENTIME_label' in df.columns:
        x = 'K2Q31B_label'
        y = 'SCREENTIME_label'
        plt.figure()
        sns.boxplot(x='K2Q31B_label', y='SCREENTIME', data=df, palette='Set2')
        plt.title(f'Screen Time (hours) by ADHD Diagnosis (Currently)')
        plt.xlabel('ADHD Diagnosis (Currently): Yes/No')
        plt.ylabel('Screen Time (hours category)')
        plt.tight_layout()
        plt.savefig('plots/bivariate_ADHD_SCREENTIME.png')
        plt.close()
    else:
        plt.figure()
        sns.boxplot(x='K2Q31B', y='SCREENTIME', data=df, palette='Set2')
        plt.title(f'{st_desc} by {adhd_desc}')
        plt.xlabel(adhd_desc)
        plt.ylabel(st_desc)
        plt.tight_layout()
        plt.savefig('plots/bivariate_ADHD_SCREENTIME.png')
        plt.close()
    print(f'Visual: See plots/bivariate_ADHD_SCREENTIME.png. This boxplot compares {st_desc} between groups defined by {adhd_desc}. {adhd_q} {st_q}')
    print('Interpretation: If one group has higher/lower screen time, this may suggest an association with ADHD risk.')


# --- Multivariate analysis: Logistic regression for ADHD (K2Q31B) vs. screen time and sex ---
if all(col in df.columns for col in ['K2Q31B', 'SCREENTIME', 'SC_SEX']):
    print('\nMultivariate analysis: Predicting ADHD diagnosis (currently) from screen time and sex')
    # Prepare data: drop NA, recode
    dff = df[['K2Q31B', 'SCREENTIME', 'SC_SEX']].dropna()
    dff['ADHD'] = dff['K2Q31B'].map({1:1, 2:0})
    dff['SEX'] = dff['SC_SEX'].map({1:'Male', 2:'Female'})
    # Treat screentime as ordinal
    model = smf.logit('ADHD ~ SCREENTIME + C(SEX)', data=dff).fit(disp=0)
    print(model.summary())
    print('Interpretation: Positive coefficients mean higher screen time or being male/female increases odds of ADHD. Significant p-values (<0.05) indicate a likely association.')

    # Visual: ADHD by screen time and sex (grouped bar)
    import matplotlib.ticker as mticker
    plt.figure(figsize=(8,5))
    grouped = dff.groupby(['SCREENTIME', 'SEX'])['ADHD'].mean().unstack()
    grouped.index = grouped.index.map(category_maps['SCREENTIME'])
    grouped.plot(kind='bar', ax=plt.gca(), color=['#66b3ff', '#ffb366'])
    plt.title('Proportion with ADHD by Screen Time and Sex')
    plt.xlabel('Screen Time (hours)')
    plt.ylabel('Proportion with ADHD')
    plt.legend(title='Sex')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('plots/multivariate_ADHD_SCREENTIME_SEX.png')
    plt.close()
    print('Visual: See plots/multivariate_ADHD_SCREENTIME_SEX.png. This grouped bar plot shows the proportion of children with ADHD for each screen time category, split by sex.')

# --- Multivariate: ADHD vs. sleep and screentime ---
if all(col in df.columns for col in ['K2Q31B', 'HOURSLEEP', 'SCREENTIME']):
    print('\nMultivariate visual: ADHD diagnosis by sleep hours and screen time')
    dff = df[['K2Q31B', 'HOURSLEEP', 'SCREENTIME']].dropna()
    dff['ADHD'] = dff['K2Q31B'].map({1:1, 2:0})
    # Bin sleep for heatmap
    sleep_bins = [0, 6, 8, 10, 12, 24]
    sleep_labels = ['<6', '6-8', '8-10', '10-12', '12+']
    dff['SLEEP_BIN'] = pd.cut(dff['HOURSLEEP'], bins=sleep_bins, labels=sleep_labels, right=False)
    # Heatmap: ADHD rate by sleep and screentime
    pivot = dff.pivot_table(index='SLEEP_BIN', columns='SCREENTIME', values='ADHD', aggfunc='mean')
    plt.figure(figsize=(8,5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Proportion with ADHD'})
    plt.title('Proportion with ADHD by Sleep Hours and Screen Time')
    plt.xlabel('Screen Time (hours)')
    plt.ylabel('Sleep Hours (binned)')
    plt.xticks(ticks=np.arange(0.5,5.5), labels=[category_maps['SCREENTIME'][i] for i in range(1,6)], rotation=0)
    plt.tight_layout()
    plt.savefig('plots/multivariate_ADHD_SLEEP_SCREENTIME.png')
    plt.close()
    print('Visual: See plots/multivariate_ADHD_SLEEP_SCREENTIME.png. This heatmap shows the proportion of children with ADHD for each combination of sleep hours and screen time.')

# --- Correlation heatmap (already present, but ensure saved and explained) ---
if len(num_vars) > 1:
    corr = df[num_vars].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='bwr', center=0, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix of Numeric Variables')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    print('Visual: See plots/correlation_heatmap.png. This heatmap shows the strength and direction of relationships between numeric variables. Red = negative, blue = positive.')

# --- Explain difference between K2Q31A and K2Q31B visuals ---
print('\n--- Difference between K2Q31A and K2Q31B Visuals ---')
print('K2Q31A: "Has a doctor or other health care provider EVER told you that this child has Attention Deficit Disorder (ADD) or Attention-Deficit/Hyperactivity Disorder (ADHD)?"')
print('K2Q31B: "If yes, does this child CURRENTLY have ADD or ADHD?"')
print('Visuals for K2Q31A show the lifetime prevalence of ADHD diagnosis, while K2Q31B visuals show current diagnosis. Differences in the plots reflect children who may have been diagnosed in the past but are not currently considered to have ADHD.')

# 4. Missing Data Pattern

plt.figure(figsize=(10,4))
sns.heatmap(df.isnull(), cbar=False)
plt.title('Missing Data Heatmap')
plt.tight_layout()
plt.savefig('plots/missing_data_heatmap.png')
plt.close()
print('Visual: See plots/missing_data_heatmap.png. This heatmap shows where data is missing across all variables.')
print('Interpretation: Large blocks of missingness may bias results or require imputation.')

# 5. Correlation (numeric only)

print('\nCorrelation matrix:')

print(df[num_vars].corr())
plt.figure(figsize=(8,6))
sns.heatmap(df[num_vars].corr(), annot=True, cmap='bwr', center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix (Red=Negative, Blue=Positive)')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()
print('Visual: See plots/correlation_matrix.png. This heatmap shows positive (blue) and negative (red) linear relationships between numeric variables. Values near +1 are strong positive, near -1 are strong negative, and near 0 are weak or no correlation.')
print('Interpretation: Strong correlations may indicate redundancy, confounding, or important relationships for modeling.')

# 6. Stratified ADHD prevalence by sex

if 'K2Q31B' in df.columns and 'SC_SEX' in df.columns:
    adhd_desc, adhd_q = var_dict.get('K2Q31B', ('K2Q31B', ''))
    sex_desc, sex_q = var_dict.get('SC_SEX', ('SC_SEX', ''))
    print(f'\n{adhd_desc} by {sex_desc}:')
    print(pd.crosstab(df['SC_SEX'], df['K2Q31B'], normalize='index'))
    plt.figure()
    pd.crosstab(df['SC_SEX'], df['K2Q31B'], normalize='index').plot(kind='bar', stacked=True)
    plt.title(f'{adhd_desc} by {sex_desc}')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig('plots/adhd_by_sex.png')
    plt.close()
    print(f'Visual: See plots/adhd_by_sex.png. This bar plot shows the proportion of {adhd_desc} by {sex_desc}. {adhd_q} {sex_q}')
    print('Interpretation: Differences may suggest sex is a risk factor or confounder for ADHD.')

# 7. Outlier check for screen time

if 'SCREENTIME' in df.columns:
    st_desc, st_q = var_dict.get('SCREENTIME', ('SCREENTIME', ''))
    plt.figure()
    sns.boxplot(x=df['SCREENTIME'])
    plt.title(f'{st_desc} Outlier Check')
    plt.xlabel(st_desc)
    plt.tight_layout()
    plt.savefig('plots/screentime_outlier.png')
    plt.close()
    print(f'Visual: See plots/screentime_outlier.png. This boxplot highlights extreme values in {st_desc}. {st_q}')
    print('Interpretation: Outliers may need to be investigated or handled in modeling.')

print('\nEDA complete. All plots saved in the plots/ directory. See comments in the script for interpretation guidance.')
