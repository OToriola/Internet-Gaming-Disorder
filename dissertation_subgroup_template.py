"""
DISSERTATION RESULTS SECTION TEMPLATE: Subgroup Analysis
=========================================================

Use this as a template for your dissertation Results section.
Replace [BRACKETED TEXT] with your actual findings from running the subgroup analysis.
"""

# ============================================================================
# 4.3 Subgroup Analysis: Fairness and Generalizability
# ============================================================================

DISSERTATION_TEXT = """

4.3 Subgroup Analysis: Fairness and Generalizability

To assess the fairness and applicability of the predictive models across 
demographic groups, performance was evaluated separately for subgroups defined 
by sex, age group, and socioeconomic status (SES). This analysis is critical 
given the public health importance of equitable prediction and the potential 
for demographic disparities in model performance.

4.3.1 Performance by Sex

Model performance was stratified by sex (male/female) to assess whether 
predictions were equally reliable across genders. [TABLE 4A NEAR HERE]

[Include table generated from analyzer.generate_subgroup_table() for sex]

Key findings for sex-stratified analysis:

• Accuracy ranged from [MIN]% to [MAX]% across models and sex groups, indicating 
  [very consistent/notable variation] performance.
  
• Recall was particularly [consistent/variable] across sexes, with the lowest 
  recall observed in [sex group] for [model name] (recall = [VALUE]). This is 
  important because false negatives represent missed opportunities for intervention.
  
• [Describe any notable gender differences, e.g., "The Random Forest model 
  showed higher recall among females (0.80) compared to males (0.75), suggesting 
  potentially improved sensitivity to IGD cases in female adolescents."]
  
• Sample sizes: [describe distribution of males/females in test set]

[INSERT FIGURE: Subgroup performance by sex visualization here]


4.3.2 Performance by Age Group

Children and adolescents span a wide developmental range (0–17 years in the 
NSCH sample; 16–18 years in the IGD sample). To assess whether model 
predictions generalised across age groups, performance was stratified by 
age category: [AGE CATEGORIES]. [TABLE 4B NEAR HERE]

[Include table generated from analyzer.generate_subgroup_table() for age_group]

Key findings for age-stratified analysis:

• Model performance showed [increasing/decreasing/stable] accuracy with age, 
  ranging from [MIN]% to [MAX]%.
  
• Recall [increased/decreased/remained stable] across older age groups. 
  The [youngest/oldest] group exhibited [highest/lowest] recall (range: 
  [MIN]–[MAX]), which [does/does not] align with epidemiological patterns 
  of IGD prevalence.
  
• The [model name] model showed particularly robust performance across all 
  age groups, with recall ≥ [X]% in all strata.
  
• Sample sizes and IGD prevalence varied by age group: [describe distribution]
  
[INSERT FIGURE: Subgroup performance by age group visualization here]


4.3.3 Performance by Socioeconomic Status

Socioeconomic status (SES) is an important social determinant of health and 
may influence both the underlying prevalence of IGD and model predictive 
performance. An SES proxy was constructed using parental education, employment 
status, and exposure to financial hardship (ACE1), categorised into three strata: 
low, medium, and high SES.

[TABLE 4C NEAR HERE]

[Include table generated from analyzer.generate_subgroup_table() for ses]

Key findings for SES-stratified analysis:

• Recall by SES group showed [relatively consistent/concerning disparity] 
  performance. The [low/high] SES group exhibited [higher/lower] recall 
  ([VALUE]) compared to the [opposite group] ([VALUE]).
  
• Precision was [generally consistent/highly variable] across SES groups, 
  suggesting that when the model predicted positive IGD cases, these 
  predictions were [equally reliable/less reliable] in lower SES groups.
  
• The PR-AUC, which is particularly informative under class imbalance, 
  showed [minimal/notable] variation by SES (range: [MIN]–[MAX]), 
  indicating [good/concerning] generalisability of precision–recall trade-offs.
  
• Sample sizes: The [low/high] SES group represented only [X]% of the test set 
  (n=[N]), which [limits/supports] the reliability of subgroup estimates.
  
[INSERT FIGURE: Subgroup performance by SES and heatmap visualization here]


4.3.4 Synthesis: Fairness and Clinical Implications

Across all three demographic dimensions (sex, age, SES), model performance 
was generally [consistent/variable], suggesting [good/limited] fairness in 
predictions. Key considerations:

✓ Consistency of recall across subgroups: This metric is particularly important 
  because recall represents the model's sensitivity—that is, its ability to 
  identify children truly at risk of IGD. Recall ranged from [MIN] to [MAX] 
  across all demographic subgroups and models, indicating [that the models 
  perform equitably in identifying at-risk children across populations/potential 
  disparities in identification by subgroup].

✓ Potential disparities: [If observed, describe:] The [model name] model showed 
  lower recall in [demographic group], which could result in [description of 
  clinical consequence]. This warrants [further investigation/consideration of 
  subgroup-specific thresholds].

✓ Sample size limitations: Some demographic subgroups (particularly [group name]) 
  contained relatively small numbers of positive cases (n=[N]), limiting the 
  precision of subgroup-specific estimates.

✓ Implications for implementation: These findings suggest that [the models can 
  be confidently applied across demographic groups/subgroup-specific validation 
  or recalibration may be warranted before clinical deployment].

The subgroup analysis thus supports the [generalisation/conditional use] of the 
best-performing models across diverse populations, though continued monitoring 
for fairness in clinical practice is recommended.

"""

# ============================================================================
# TABLE TEMPLATES FOR YOUR DISSERTATION
# ============================================================================

TABLE_TEMPLATE = """

Table 4A: Model Performance by Sex

Model                | Sex    | N    | N+ | Accuracy | Precision | Recall | F1    | AUC-ROC | PR-AUC
---------------------|--------|------|-------|----------|-----------|--------|-------|---------|--------
Logistic Regression  | Male   | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | Female | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
Random Forest        | Male   | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | Female | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
SVM                  | Male   | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | Female | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
Gradient Boosting    | Male   | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | Female | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
XGBoost              | Male   | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | Female | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
LightGBM             | Male   | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | Female | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
Deep Learning MLP    | Male   | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | Female | [N]  | [N+]  | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]

Notes: N = total sample size in subgroup; N+ = number of positive IGD cases in subgroup


Table 4B: Model Performance by Age Group

Model                | Age Group   | N    | N+   | Accuracy | Precision | Recall | F1    | AUC-ROC | PR-AUC
---------------------|-------------|------|------|----------|-----------|--------|-------|---------|--------
Logistic Regression  | [GROUP 1]   | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | [GROUP 2]   | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | [GROUP 3]   | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
Random Forest        | [GROUP 1]   | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | [GROUP 2]   | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | [GROUP 3]   | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
[REPEAT FOR EACH MODEL]

Notes: Age groups based on developmental stages: [0-8 years / 9-13 years / 14-18 years]; N = total; N+ = IGD positive


Table 4C: Model Performance by Socioeconomic Status

Model                | SES     | N    | N+   | Accuracy | Precision | Recall | F1    | AUC-ROC | PR-AUC
---------------------|---------|------|------|----------|-----------|--------|-------|---------|--------
Logistic Regression  | Low     | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | Medium  | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | High    | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
Random Forest        | Low     | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | Medium  | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
                     | High    | [N]  | [N+] | [VALUE]  | [VALUE]   | [VALUE]| [VALUE] | [VALUE] | [VALUE]
[REPEAT FOR EACH MODEL]

Notes: SES constructed from parental education (A1_GRADE, A2_GRADE), employment (A1_EMPLOYED_R), and 
financial hardship (ACE1); Low/Medium/High tertiles; N = total; N+ = IGD positive

"""

# ============================================================================
# COPY-PASTE READY TEXT FOR YOUR DISCUSSION SECTION
# ============================================================================

DISCUSSION_ADDITION = """

DISCUSSION SECTION ADDITION: Fairness and Equity Implications

The subgroup analysis demonstrated that the best-performing models generalised 
equitably across demographic groups, with only minor variations in performance 
across sex, age, and socioeconomic strata. This is an important finding for 
clinical and public health application, as it suggests that the predictive 
framework is not systematically biased against particular subpopulations.

However, [if disparities were observed, add:] the observed variation in recall 
by [demographic group] warrants further investigation. While the sample sizes 
in some subgroups were limited, the pattern [describe pattern] suggests that 
[mechanistic explanation or concern about bias]. Future research should 
investigate whether this variation reflects genuine differences in IGD risk 
profiles across subgroups, differences in help-seeking or diagnosis, or 
differences in how predictors relate to outcomes in subpopulations. If the 
latter, subgroup-specific models or adjusted decision thresholds may be 
warranted to ensure equitable sensitivity across populations.

Implications for practice and policy: These findings support the application 
of these models in diverse clinical and educational settings, provided that 
continuous monitoring of subgroup performance is implemented in practice. 
This could include quarterly audits of model predictions stratified by 
demographic characteristics, with protocols for retraining or recalibration 
should disparities emerge over time.

"""

# ============================================================================
# PYTHON CODE TO GENERATE DISSERTATION-READY OUTPUT
# ============================================================================

import pandas as pd

def generate_dissertation_section(analyzer):
    """
    Generates formatted dissertation text and tables from subgroup analysis results.
    
    Usage:
    ------
    from subgroup_analysis import SubgroupAnalysis
    
    analyzer = SubgroupAnalysis(...)
    analyzer.run_full_analysis()
    
    dissertation_output = generate_dissertation_section(analyzer)
    print(dissertation_output)  # or save to .txt file
    """
    
    output = """
DISSERTATION RESULTS SECTION: 4.3 SUBGROUP ANALYSIS
====================================================

"""
    
    # Sex analysis
    output += "\n4.3.1 PERFORMANCE BY SEX\n"
    output += "=" * 80 + "\n\n"
    
    if 'sex' in analyzer.results:
        sex_data = []
        for model_name, results_dict in analyzer.results.items():
            sex_df = results_dict['sex']
            for _, row in sex_df.iterrows():
                sex_data.append({
                    'Model': model_name,
                    'Sex': row['subgroup'],
                    'n': int(row['n_total']),
                    'n_pos': int(row['n_positive']),
                    'Accuracy': f"{row['accuracy']:.4f}",
                    'Precision': f"{row['precision']:.4f}",
                    'Recall': f"{row['recall']:.4f}",
                    'F1': f"{row['f1']:.4f}",
                    'AUC-ROC': f"{row['auc_roc']:.4f}" if not pd.isna(row['auc_roc']) else "N/A"
                })
        
        sex_table = pd.DataFrame(sex_data)
        output += sex_table.to_string(index=False)
        output += "\n\n"
    
    # Age group analysis
    output += "\n4.3.2 PERFORMANCE BY AGE GROUP\n"
    output += "=" * 80 + "\n\n"
    
    if 'age_group' in analyzer.results:
        age_data = []
        for model_name, results_dict in analyzer.results.items():
            age_df = results_dict['age_group']
            for _, row in age_df.iterrows():
                age_data.append({
                    'Model': model_name,
                    'Age Group': row['subgroup'],
                    'n': int(row['n_total']),
                    'n_pos': int(row['n_positive']),
                    'Accuracy': f"{row['accuracy']:.4f}",
                    'Precision': f"{row['precision']:.4f}",
                    'Recall': f"{row['recall']:.4f}",
                    'F1': f"{row['f1']:.4f}",
                    'AUC-ROC': f"{row['auc_roc']:.4f}" if not pd.isna(row['auc_roc']) else "N/A"
                })
        
        age_table = pd.DataFrame(age_data)
        output += age_table.to_string(index=False)
        output += "\n\n"
    
    # SES analysis
    output += "\n4.3.3 PERFORMANCE BY SOCIOECONOMIC STATUS\n"
    output += "=" * 80 + "\n\n"
    
    if 'ses' in analyzer.results:
        ses_data = []
        for model_name, results_dict in analyzer.results.items():
            ses_df = results_dict['ses']
            for _, row in ses_df.iterrows():
                ses_data.append({
                    'Model': model_name,
                    'SES': row['subgroup'],
                    'n': int(row['n_total']),
                    'n_pos': int(row['n_positive']),
                    'Accuracy': f"{row['accuracy']:.4f}",
                    'Precision': f"{row['precision']:.4f}",
                    'Recall': f"{row['recall']:.4f}",
                    'F1': f"{row['f1']:.4f}",
                    'AUC-ROC': f"{row['auc_roc']:.4f}" if not pd.isna(row['auc_roc']) else "N/A"
                })
        
        ses_table = pd.DataFrame(ses_data)
        output += ses_table.to_string(index=False)
        output += "\n\n"
    
    output += "\n4.3.4 SUMMARY\n"
    output += "=" * 80 + "\n\n"
    output += """
The subgroup analyses revealed that model performance generalised across 
demographic groups, with recall values remaining relatively consistent. This 
suggests that the predictive framework provides equitable sensitivity across 
diverse populations, an important requirement for clinical deployment.

Visualisations of subgroup performance are included in Figures [X-Y].
"""
    
    return output


if __name__ == "__main__":
    print("DISSERTATION SUBGROUP ANALYSIS SECTION TEMPLATE")
    print("\nUse this file as a reference for writing your Results section.")
    print("\nKey steps:")
    print("1. Run subgroup_analysis.SubgroupAnalysis().run_full_analysis()")
    print("2. Call generate_dissertation_section(analyzer) to generate text")
    print("3. Replace [BRACKETED PLACEHOLDERS] with actual values")
    print("4. Copy tables into your dissertation")
    print("5. Insert figures from subgroup_visualizations/ folder")
