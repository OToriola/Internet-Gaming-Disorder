"""
Subgroup Analysis for IGD Predictive Models
============================================

This script performs fairness and applicability analysis by evaluating model performance
across demographic subgroups (sex, age group, socioeconomic status) as mentioned in the methodology.

It should be integrated after model training and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

class SubgroupAnalysis:
    """
    Evaluates model performance across demographic subgroups to assess fairness and generalizability.
    """
    
    def __init__(self, X_test, y_test, y_pred_dict, y_prob_dict, demographic_data):
        """
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test set features
        y_test : np.array or pd.Series
            True test labels
        y_pred_dict : dict
            Dictionary with model names as keys and predicted labels as values
        y_prob_dict : dict
            Dictionary with model names as keys and predicted probabilities as values
        demographic_data : pd.DataFrame
            Demographic information with columns: 'sex', 'age_group', 'ses'
            Must have same index as X_test and y_test
        """
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred_dict = y_pred_dict
        self.y_prob_dict = y_prob_dict
        self.demographics = demographic_data
        
        # Verify indices match
        assert len(self.y_test) == len(self.demographics), "Demographics length must match test set"
        
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate all performance metrics for a given prediction set"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            except:
                metrics['auc_roc'] = np.nan
                metrics['pr_auc'] = np.nan
        else:
            metrics['auc_roc'] = np.nan
            metrics['pr_auc'] = np.nan
            
        return metrics
    
    def analyze_by_sex(self, model_name, y_pred, y_prob=None):
        """Analyze performance by sex (Male/Female/Other)"""
        subgroup_results = []
        
        for sex in self.demographics['sex'].unique():
            if pd.isna(sex):
                continue
                
            mask = self.demographics['sex'] == sex
            y_true_sub = self.y_test[mask]
            y_pred_sub = y_pred[mask]
            y_prob_sub = y_prob[mask] if y_prob is not None else None
            
            metrics = self.calculate_metrics(y_true_sub, y_pred_sub, y_prob_sub)
            metrics['subgroup'] = str(sex)
            metrics['n_total'] = mask.sum()
            metrics['n_positive'] = (y_true_sub == 1).sum()
            metrics['positive_rate'] = metrics['n_positive'] / metrics['n_total'] if metrics['n_total'] > 0 else 0
            
            subgroup_results.append(metrics)
        
        return pd.DataFrame(subgroup_results)
    
    def analyze_by_age_group(self, model_name, y_pred, y_prob=None):
        """Analyze performance by age group"""
        subgroup_results = []
        
        # Sort age groups in logical order if they're categorical
        age_groups = self.demographics['age_group'].unique()
        try:
            age_groups = sorted(age_groups)
        except:
            pass
        
        for age_group in age_groups:
            if pd.isna(age_group):
                continue
                
            mask = self.demographics['age_group'] == age_group
            y_true_sub = self.y_test[mask]
            y_pred_sub = y_pred[mask]
            y_prob_sub = y_prob[mask] if y_prob is not None else None
            
            metrics = self.calculate_metrics(y_true_sub, y_pred_sub, y_prob_sub)
            metrics['subgroup'] = str(age_group)
            metrics['n_total'] = mask.sum()
            metrics['n_positive'] = (y_true_sub == 1).sum()
            metrics['positive_rate'] = metrics['n_positive'] / metrics['n_total'] if metrics['n_total'] > 0 else 0
            
            subgroup_results.append(metrics)
        
        return pd.DataFrame(subgroup_results)
    
    def analyze_by_ses(self, model_name, y_pred, y_prob=None):
        """Analyze performance by socioeconomic status"""
        subgroup_results = []
        
        # Sort SES categories in logical order if possible
        ses_groups = self.demographics['ses'].unique()
        ses_order = {'Low': 0, 'Medium': 1, 'High': 2}
        try:
            ses_groups = sorted(ses_groups, key=lambda x: ses_order.get(str(x), 999))
        except:
            pass
        
        for ses in ses_groups:
            if pd.isna(ses):
                continue
                
            mask = self.demographics['ses'] == ses
            y_true_sub = self.y_test[mask]
            y_pred_sub = y_pred[mask]
            y_prob_sub = y_prob[mask] if y_prob is not None else None
            
            metrics = self.calculate_metrics(y_true_sub, y_pred_sub, y_prob_sub)
            metrics['subgroup'] = str(ses)
            metrics['n_total'] = mask.sum()
            metrics['n_positive'] = (y_true_sub == 1).sum()
            metrics['positive_rate'] = metrics['n_positive'] / metrics['n_total'] if metrics['n_total'] > 0 else 0
            
            subgroup_results.append(metrics)
        
        return pd.DataFrame(subgroup_results)
    
    def run_full_analysis(self):
        """
        Run complete subgroup analysis across all models and demographic factors.
        
        Returns:
        --------
        results_dict : dict
            Dictionary containing subgroup results for each model and demographic variable
        """
        all_results = {}
        
        for model_name in self.y_pred_dict.keys():
            print(f"\n{'='*70}")
            print(f"Processing: {model_name}")
            print(f"{'='*70}")
            
            y_pred = self.y_pred_dict[model_name]
            y_prob = self.y_prob_dict.get(model_name, None)
            
            model_results = {}
            
            # Sex analysis
            print("\nAnalyzing performance by SEX...")
            sex_results = self.analyze_by_sex(model_name, y_pred, y_prob)
            model_results['sex'] = sex_results
            print(sex_results.to_string(index=False))
            
            # Age group analysis
            print("\nAnalyzing performance by AGE GROUP...")
            age_results = self.analyze_by_age_group(model_name, y_pred, y_prob)
            model_results['age_group'] = age_results
            print(age_results.to_string(index=False))
            
            all_results[model_name] = model_results
        
        self.results = all_results
        return all_results
    
    def generate_subgroup_table(self, output_format='latex'):
        """
        Generate formatted tables suitable for dissertation inclusion.
        
        Parameters:
        -----------
        output_format : str, default='latex'
            'latex' for LaTeX table format (suitable for Word)
            'markdown' for Markdown table format
            'csv' for CSV format
        """
        if not self.results:
            print("Run run_full_analysis() first")
            return
        
        tables = {}
        
        for demographic in ['sex', 'age_group']:
            print(f"\n{'='*80}")
            print(f"TABLE: Subgroup Performance Analysis by {demographic.upper()}")
            print(f"{'='*80}\n")
            
            combined_results = []
            
            for model_name, model_results in self.results.items():
                df = model_results[demographic].copy()
                df.insert(0, 'Model', model_name)
                combined_results.append(df)
            
            combined_df = pd.concat(combined_results, ignore_index=True)
            
            # Round numeric columns for readability
            numeric_cols = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'pr_auc', 'positive_rate']
            for col in numeric_cols:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].round(4)
            
            if output_format == 'latex':
                latex_table = combined_df.to_latex(index=False)
                print(latex_table)
                tables[demographic] = latex_table
                
            elif output_format == 'markdown':
                markdown_table = combined_df.to_markdown(index=False)
                print(markdown_table)
                tables[demographic] = markdown_table
                
            else:  # csv
                print(combined_df.to_csv(index=False))
                tables[demographic] = combined_df.to_csv(index=False)
        
        return tables
    
    def visualize_subgroup_performance(self, save_dir='subgroup_visualizations'):
        """
        Create visualizations of performance across subgroups.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save visualization files
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.results:
            print("Run run_full_analysis() first")
            return
        
        metrics_to_plot = ['accuracy', 'recall', 'f1', 'auc_roc']
        
        # Create subplots for each demographic variable (sex and age_group only)
        for demographic in ['sex', 'age_group']:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx]
                
                data_points = []
                model_names = []
                subgroup_names = []
                
                for model_name, model_results in self.results.items():
                    df = model_results[demographic]
                    
                    for _, row in df.iterrows():
                        data_points.append({
                            'Model': model_name,
                            'Subgroup': row['subgroup'],
                            metric: row[metric]
                        })
                
                plot_df = pd.DataFrame(data_points)
                
                # Create grouped bar plot
                plot_df_pivot = plot_df.pivot(index='Subgroup', columns='Model', values=metric)
                plot_df_pivot.plot(kind='bar', ax=ax, width=0.8)
                
                ax.set_title(f'{metric.upper()} by {demographic.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
                ax.set_xlabel(demographic.replace('_', ' ').title(), fontsize=10)
                ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.grid(axis='y', alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            filepath = os.path.join(save_dir, f'subgroup_performance_{demographic}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            plt.close()
        
        # Create heatmap of recall by subgroup (important for fairness)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, demographic in enumerate(['sex', 'age_group']):
            data_for_heatmap = []
            
            for model_name, model_results in self.results.items():
                df = model_results[demographic]
                row = []
                subgroups = df['subgroup'].tolist()
                
                for _, record in df.iterrows():
                    row.append(record['recall'])
                
                data_for_heatmap.append(row)
            
            model_names = list(self.results.keys())
            heatmap_df = pd.DataFrame(data_for_heatmap, 
                                     index=model_names,
                                     columns=subgroups)
            
            sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=axes[idx], cbar_kws={'label': 'Recall'}, vmin=0, vmax=1)
            axes[idx].set_title(f'Recall by {demographic.replace("_", " ").title()}', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Model', fontsize=10)
            axes[idx].set_xlabel(demographic.replace('_', ' ').title(), fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(save_dir, 'subgroup_recall_heatmaps.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()


def create_demographic_data_from_features(X_test, df_full):
    """
    Helper function to extract demographic data from your feature dataframe.
    
    Customize based on your actual column names:
    - Sex/Gender column: Usually named 'sex', 'gender', 'SC_SEX', etc.
    - Age column: Usually named 'age', 'age_years', 'SC_AGE_YEARS', etc.
    - SES column: Derived from education, employment, parental education, etc.
    
    Returns:
    --------
    demographics : pd.DataFrame
        DataFrame with columns: 'sex', 'age_group', 'ses'
    """
    
    demographics = pd.DataFrame(index=X_test.index)
    
    # Extract sex (customize column name based on your data)
    if 'SC_SEX' in df_full.columns:
        demographics['sex'] = df_full['SC_SEX'].map({1: 'Male', 2: 'Female'})
    elif 'sex' in df_full.columns:
        demographics['sex'] = df_full['sex']
    
    # Create age groups (customize based on your age column)
    if 'SC_AGE_YEARS' in df_full.columns:
        age_col = 'SC_AGE_YEARS'
    elif 'age' in df_full.columns:
        age_col = 'age'
    else:
        age_col = None
    
    if age_col:
        demographics['age_group'] = pd.cut(df_full[age_col], 
                                          bins=[0, 8, 13, 18, 100],
                                          labels=['0-8 years', '9-13 years', '14-18 years', '18+ years'])
    
    # Create SES proxy (customize based on your available indicators)
    # This is a simple example - adjust based on your actual SES indicators
    ses_score = 0
    
    if 'A1_GRADE' in df_full.columns or 'A2_GRADE' in df_full.columns:
        # Parent education
        edu_cols = [col for col in df_full.columns if 'GRADE' in col]
        ses_score += df_full[edu_cols].fillna(0).mean(axis=1)
    
    if 'A1_EMPLOYED_R' in df_full.columns:
        # Employment status (1=employed is good)
        ses_score += df_full['A1_EMPLOYED_R'].fillna(0)
    
    if 'ACE1' in df_full.columns:
        # Invert ACE1 (financial hardship) - higher ACE1 = lower SES
        ses_score -= df_full['ACE1'].fillna(0)
    
    # Categorize SES into Low/Medium/High
    if ses_score.std() > 0:
        demographics['ses'] = pd.qcut(ses_score, q=3, labels=['Low', 'Medium', 'High'])
    else:
        demographics['ses'] = 'Unknown'
    
    return demographics


if __name__ == "__main__":
    """
    Example usage:
    
    # After you've trained your models and have predictions:
    
    # 1. Create demographic data from your test set
    demographics = create_demographic_data_from_features(X_test, df_original)
    
    # 2. Initialize the subgroup analyzer
    analyzer = SubgroupAnalysis(
        X_test=X_test,
        y_test=y_test,
        y_pred_dict=y_pred_dict,  # {'Model_name': y_pred_array, ...}
        y_prob_dict=y_prob_dict,   # {'Model_name': y_prob_array, ...}
        demographic_data=demographics
    )
    
    # 3. Run analysis
    results = analyzer.run_full_analysis()
    
    # 4. Generate tables for dissertation
    tables = analyzer.generate_subgroup_table(output_format='latex')
    
    # 5. Create visualizations
    analyzer.visualize_subgroup_performance(save_dir='subgroup_visualizations')
    """
    
    print("Subgroup Analysis Module")
    print("See docstrings and example usage above")
