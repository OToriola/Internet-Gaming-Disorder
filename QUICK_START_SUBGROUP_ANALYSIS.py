"""
QUICK START GUIDE: Adding Subgroup Analysis to Your Dissertation
==================================================================

This is the fastest way to get subgroup analysis working with your existing code.
"""

# ============================================================================
# OPTION 1: STANDALONE USAGE (EASIEST - Start here!)
# ============================================================================

"""
If you already have saved predictions from your models, use this approach:

1. Make sure you have your predictions saved somewhere, like:
   - y_test: true labels
   - y_pred_dict: {'model_name': predictions_array, ...}
   - y_prob_dict: {'model_name': probabilities_array, ...}
   - A dataframe with demographic columns: sex, age_group, ses

2. Run this code:
"""

import pandas as pd
from subgroup_analysis import SubgroupAnalysis

# Example: If your data is in CSV files
X_test = pd.read_csv('X_test.csv', index_col=0)
y_test = pd.read_csv('y_test.csv', index_col=0).squeeze()
demographics = pd.read_csv('demographics.csv', index_col=0)

# If your predictions are saved in separate files
import pickle
with open('predictions.pkl', 'rb') as f:
    y_pred_dict, y_prob_dict = pickle.load(f)

# Run analysis
analyzer = SubgroupAnalysis(X_test, y_test, y_pred_dict, y_prob_dict, demographics)
results = analyzer.run_full_analysis()

# Generate tables for dissertation
tables = analyzer.generate_subgroup_table(output_format='markdown')

# Create visualizations
analyzer.visualize_subgroup_performance()

# Print dissertation-ready text
from dissertation_subgroup_template import generate_dissertation_section
dissertation_text = generate_dissertation_section(analyzer)
print(dissertation_text)

# Save to file
with open('RESULTS_SUBGROUP_ANALYSIS.txt', 'w') as f:
    f.write(dissertation_text)


# ============================================================================
# OPTION 2: INTEGRATE INTO EXISTING ml_prediction_demo.py
# ============================================================================

"""
If you want to add this to your ml_prediction_demo.py:

Step A: Add to imports (line ~1-20 of ml_prediction_demo.py):
"""

from subgroup_analysis import SubgroupAnalysis, create_demographic_data_from_features
from dissertation_subgroup_template import generate_dissertation_section


"""
Step B: In your load_and_preprocess_data() function, also return the full dataframe:
"""

def load_and_preprocess_data():
    """Modified to also return full dataframe for demographic extraction"""
    df = pd.read_excel("IGD Database.xlsx")
    
    # [Your existing preprocessing code...]
    
    # At the end, return both processed AND original data:
    return processed_X, processed_y, df  # <-- Add df here


"""
Step C: In your evaluate_models() function (or main section), add this after model training:
"""

# After you have trained all models and have y_test, y_pred_dict, y_prob_dict:

# Extract demographics from test set
df_test = df.loc[X_test.index]  # Get the original data for test indices
demographics = create_demographic_data_from_features(X_test, df_test)

# Run subgroup analysis
analyzer = SubgroupAnalysis(X_test, y_test, y_pred_dict, y_prob_dict, demographics)
subgroup_results = analyzer.run_full_analysis()

# Generate dissertation output
tables = analyzer.generate_subgroup_table(output_format='markdown')
analyzer.visualize_subgroup_performance(save_dir='subgroup_visualizations')

dissertation_output = generate_dissertation_section(analyzer)
with open('RESULTS_4_3_SUBGROUP_ANALYSIS.txt', 'w') as f:
    f.write(dissertation_output)

print("\n✓ Subgroup analysis complete!")
print("✓ Tables saved to RESULTS_4_3_SUBGROUP_ANALYSIS.txt")
print("✓ Visualizations saved to subgroup_visualizations/")


# ============================================================================
# CUSTOMIZE FOR YOUR SPECIFIC DATA
# ============================================================================

"""
The key customization step is telling the code what your demographic columns are.

Edit this in subgroup_analysis.py, function: create_demographic_data_from_features()

Find this section and customize the column names:
"""

def create_demographic_data_from_features(X_test, df_full):
    """
    CUSTOMIZE THESE COLUMN NAMES TO MATCH YOUR DATA:
    """
    
    demographics = pd.DataFrame(index=X_test.index)
    
    # ===== CUSTOMIZE THESE =====
    
    # Sex/Gender column name - what's it called in your data?
    # Common names: 'sex', 'gender', 'SC_SEX', 'Gender', 'gend'
    SEX_COLUMN = 'SC_SEX'  # <-- CHANGE THIS
    
    # How are males/females coded? Common patterns:
    # 1='Male', 2='Female'  OR  'M'='Male', 'F'='Female'  OR  1='Male', 0='Female'
    SEX_MAP = {1: 'Male', 2: 'Female'}  # <-- CHANGE THIS IF NEEDED
    
    if SEX_COLUMN in df_full.columns:
        demographics['sex'] = df_full[SEX_COLUMN].map(SEX_MAP)
    
    
    # Age column name - what's it called in your data?
    # Common names: 'age', 'SC_AGE_YEARS', 'age_years', 'Age'
    AGE_COLUMN = 'SC_AGE_YEARS'  # <-- CHANGE THIS
    
    if AGE_COLUMN in df_full.columns:
        # Define age bins (customize these if needed)
        demographics['age_group'] = pd.cut(
            df_full[AGE_COLUMN], 
            bins=[0, 8, 13, 18, 100],
            labels=['0-8 years', '9-13 years', '14-18 years', '18+ years']
        )
    
    
    # SES - this is more complex. Use available columns:
    # Common approach: combine education, employment, financial hardship
    SES_COLUMNS = ['A1_GRADE', 'A2_GRADE', 'A1_EMPLOYED_R', 'ACE1']
    
    ses_score = 0
    for col in SES_COLUMNS:
        if col in df_full.columns:
            ses_score = ses_score + df_full[col].fillna(0)
    
    if ses_score.std() > 0:
        demographics['ses'] = pd.qcut(ses_score, q=3, labels=['Low', 'Medium', 'High'])
    else:
        demographics['ses'] = 'Unknown'
    
    return demographics


# ============================================================================
# IF YOU'RE USING NSCH DATA SPECIFICALLY
# ============================================================================

"""
For NSCH dataset, use these customizations:
"""

# Column names for NSCH:
# Sex: 'SC_SEX' (1=Male, 2=Female)
# Age: 'SC_AGE_YEARS'
# SES components: 'A1_GRADE', 'A2_GRADE', 'A1_EMPLOYED_R', 'ACE1'

def create_demographic_data_nsch(X_test, df_full):
    """Specific for NSCH dataset"""
    demographics = pd.DataFrame(index=X_test.index)
    
    # Sex
    demographics['sex'] = df_full['SC_SEX'].map({1: 'Male', 2: 'Female'})
    
    # Age groups (0-17 range in NSCH)
    demographics['age_group'] = pd.cut(
        df_full['SC_AGE_YEARS'],
        bins=[0, 6, 12, 17],
        labels=['0-6 years', '7-12 years', '13-17 years']
    )
    
    # SES (combine parental education, employment, financial hardship)
    ses_score = (
        df_full['A1_GRADE'].fillna(0) +
        df_full['A2_GRADE'].fillna(0) +
        df_full['A1_EMPLOYED_R'].fillna(0) -
        df_full['ACE1'].fillna(0)  # Invert: higher ACE1 = lower SES
    )
    
    demographics['ses'] = pd.qcut(ses_score, q=3, labels=['Low', 'Medium', 'High'])
    
    return demographics


# ============================================================================
# IF YOU'RE USING IGD DATA SPECIFICALLY
# ============================================================================

"""
For IGD dataset, you may not have all demographic variables.
Adapt as needed:
"""

def create_demographic_data_igd(X_test, df_full):
    """Specific for IGD dataset"""
    demographics = pd.DataFrame(index=X_test.index)
    
    # Assuming IGD data has Age column (usually 16-18 years range)
    # Customize column name if different
    if 'Age' in df_full.columns:
        demographics['sex'] = df_full['Gender'].map({1: 'Male', 2: 'Female'})  # Adjust mapping
        
        demographics['age_group'] = pd.cut(
            df_full['Age'],
            bins=[14, 16, 18, 20],
            labels=['14-15', '16-17', '18+']
        )
    
    # SES might not be available in IGD dataset
    # Use a proxy or mark as unavailable
    demographics['ses'] = 'Not Available'
    
    return demographics


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
PROBLEM 1: "KeyError: column 'SC_SEX' not found"
SOLUTION: Check your actual column names
>>> df.columns
>>> df.head()  # Look at first few rows

Then update SEX_COLUMN, AGE_COLUMN, etc. with correct names


PROBLEM 2: "ValueError: Cannot cut with NaN values"
SOLUTION: The age column has missing values. Add fillna():
>>> demographics['age_group'] = pd.cut(
        df_full['SC_AGE_YEARS'].fillna(df_full['SC_AGE_YEARS'].median()),
        bins=[0, 8, 13, 18, 100],
        labels=['0-8 years', '9-13 years', '14-18 years', '18+ years']
    )


PROBLEM 3: "No subgroups have positive cases"
SOLUTION: Some subgroups may have no IGD-positive cases. This is normal.
The code handles this with zero_division=0 in precision/recall calculations.
Just verify the results make sense.


PROBLEM 4: "LightGBM or Deep Learning model not in results"
SOLUTION: Make sure you're passing ALL model predictions to y_pred_dict
>>> y_pred_dict = {
        'Logistic Regression': y_pred_lr,
        'Random Forest': y_pred_rf,
        'SVM': y_pred_svm,
        'Gradient Boosting': y_pred_gb,
        'XGBoost': y_pred_xgb,
        'LightGBM': y_pred_lgbm,
        'Deep Learning MLP': y_pred_mlp,
    }
"""


# ============================================================================
# EXPECTED OUTPUT
# ============================================================================

"""
When you run the subgroup analysis, you'll see:

CONSOLE OUTPUT (printed to screen):
======================================================================
Logistic Regression
======================================================================

Analyzing performance by SEX...
    subgroup  accuracy  precision    recall  f1-score  auc_roc  n_total  n_positive  positive_rate
Male      0.9850     0.8000  1.0000    0.8889    1.0000       52             4          0.0769
Female    0.9841     0.8000  1.0000    0.8889    1.0000       63             5          0.0794

Analyzing performance by AGE GROUP...
...

CREATED FILES:
- RESULTS_4_3_SUBGROUP_ANALYSIS.txt (copy into dissertation Results section)
- subgroup_visualizations/
  - subgroup_performance_sex.png
  - subgroup_performance_age_group.png
  - subgroup_performance_ses.png
  - subgroup_recall_heatmaps.png

These visualizations show bars comparing model performance across subgroups,
plus heatmaps of recall (the most important metric for identifying at-risk children).
"""


# ============================================================================
# NEXT STEPS FOR DISSERTATION
# ============================================================================

"""
1. Run the code above (OPTION 1 or 2)

2. Open RESULTS_4_3_SUBGROUP_ANALYSIS.txt and copy into your dissertation

3. Insert visualizations:
   - From subgroup_visualizations/ folder
   - Insert as Figures 4A, 4B, 4C (or appropriate numbering)
   - Add captions below each figure

4. Write brief interpretation:
   - Copy the template text from dissertation_subgroup_template.py
   - Replace [BRACKETED PLACEHOLDERS] with your actual values
   - Add discussion of what results mean for fairness/applicability

5. Add to your discussion section:
   - Copy DISCUSSION_ADDITION from dissertation_subgroup_template.py
   - Discuss implications for clinical use across demographic groups

DONE! Your subgroup analysis section will be complete.
"""

print(__doc__)
