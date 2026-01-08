import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Constants for features related to IGD prediction
SCREEN_TIME_VARS = ['SCREENTIME']

# Dictionary mapping variable codes to descriptive labels
EMOTIONAL_VARS_LABELS = {
    'K2Q31A': 'ADD/ADHD Diagnosis',
    'K2Q31B': 'Current ADD/ADHD',
    'K2Q31C': 'ADD/ADHD Severity',
    'K2Q31D': 'ADD/ADHD Medication',
    'ADDTREAT': 'ADD/ADHD Treatment'
}

SOCIAL_VARS_LABELS = {
    'K2Q33A': 'Anxiety Diagnosis',
    'K2Q33B': 'Current Anxiety',
    'K2Q33C': 'Anxiety Severity',
    'K2Q32A': 'Depression Diagnosis',
    'K2Q32B': 'Current Depression',
    'K2Q32C': 'Depression Severity',
    'K2Q34A': 'Behavior Problems Diagnosis',
    'K2Q34B': 'Current Behavior Problems',
    'K2Q34C': 'Behavior Problems Severity'
}

# Context variables
CONTEXT_VARS_LABELS = {
    'FAMILY_R': 'Family Structure',
    'A1_GRADE': 'Parent 1 Education',
    'A2_GRADE': 'Parent 2 Education'
}

SLEEP_VARS = ['HOURSLEEP', 'HOURSLEEP05']

# Define variable groups
EMOTIONAL_VARS = list(EMOTIONAL_VARS_LABELS.keys())
SOCIAL_VARS = list(SOCIAL_VARS_LABELS.keys())
CONTEXT_VARS = list(CONTEXT_VARS_LABELS.keys())

def load_nsch_data():
    """Load and preprocess NSCH dataset."""
    # Load the original NSCH dataset for sex information
    df_original = pd.read_excel("nsch_2023e_topical.xlsx")
    
    # Load our reduced dataset
    df = pd.read_excel("nsch2023_merged_reduced.xlsx")
    
    # Add sex from the original dataset
    if 'SC_SEX' in df_original.columns:
        df['SC_SEX'] = df_original['SC_SEX']
    
    # Combine all relevant features
    selected_vars = (['SC_SEX'] if 'SC_SEX' in df.columns else []) + SCREEN_TIME_VARS + EMOTIONAL_VARS + SOCIAL_VARS + CONTEXT_VARS + SLEEP_VARS
    
    # Select only necessary columns
    df_selected = df[selected_vars].copy()
    
    return df_selected

def create_igd_risk_score(df):
    """
    Create a comprehensive IGD risk score based on multiple domains:
    1. Screen Time (30%)
    2. ADD/ADHD & Emotional (25%)
    3. Social/Behavioral (25%)
    4. Sleep & Context (20%)
    """
    df_numeric = df.copy()
    
    # 1. Screen Time Component (30%)
    screen_time = pd.to_numeric(df_numeric['SCREENTIME'], errors='coerce')
    screen_time = screen_time.fillna(screen_time.mean())
    screen_z = StandardScaler().fit_transform(screen_time.values.reshape(-1, 1)).flatten()
    
    # 2. ADD/ADHD & Emotional Component (25%)
    emotional_data = df_numeric[EMOTIONAL_VARS].apply(pd.to_numeric, errors='coerce')
    emotional_data = emotional_data.fillna(emotional_data.mean())
    emotional_z = StandardScaler().fit_transform(emotional_data)
    emotional_score = emotional_z.mean(axis=1)
    
    # 3. Social/Behavioral Component (25%)
    social_data = df_numeric[SOCIAL_VARS].apply(pd.to_numeric, errors='coerce')
    social_data = social_data.fillna(social_data.mean())
    social_z = StandardScaler().fit_transform(social_data)
    social_score = social_z.mean(axis=1)
    
    # 4. Sleep & Context Component (20%)
    sleep_context_vars = SLEEP_VARS + CONTEXT_VARS
    context_data = df_numeric[sleep_context_vars].apply(pd.to_numeric, errors='coerce')
    context_data = context_data.fillna(context_data.mean())
    context_z = StandardScaler().fit_transform(context_data)
    context_score = context_z.mean(axis=1)
    
    # Calculate final weighted risk score
    weights = {
        'screen': 0.30,
        'emotional': 0.25,
        'social': 0.25,
        'context': 0.20
    }
    
    risk_score = (
        weights['screen'] * screen_z +
        weights['emotional'] * emotional_score +
        weights['social'] * social_score +
        weights['context'] * context_score
    )
    
    return risk_score

def analyze_risk_factors(df):
    """Analyze relationships between variables and IGD risk."""
    # Handle missing values
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Calculate IGD risk score
    df['igd_risk_score'] = create_igd_risk_score(df)
    
    # Create visualizations directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Screen Time vs Sleep Analysis
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='SCREENTIME', y='HOURSLEEP', alpha=0.5)
    plt.title('Relationship Between Screen Time and Sleep Duration', fontsize=12, pad=20)
    plt.xlabel('Screen Time (hours per day)', fontsize=10)
    plt.ylabel('Sleep Duration (hours)', fontsize=10)
    
    # Add trend line
    clean_data = df.dropna(subset=['SCREENTIME', 'HOURSLEEP'])
    z = np.polyfit(clean_data['SCREENTIME'], clean_data['HOURSLEEP'], 1)
    p = np.poly1d(z)
    plt.plot(clean_data['SCREENTIME'], p(clean_data['SCREENTIME']), "r--", alpha=0.8)
    
    # Add correlation coefficient
    corr = df['SCREENTIME'].corr(df['HOURSLEEP'])
    plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
             transform=plt.gca().transAxes, 
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig('visualizations/screen_time_vs_sleep.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Distribution by Sex (if available)
    if 'SC_SEX' in df.columns:
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Screen Time Distribution by Sex
        sns.boxplot(data=df, x='SC_SEX', y='SCREENTIME', ax=ax1)
        ax1.set_title('Screen Time Distribution by Sex')
        ax1.set_xlabel('Sex (1=Male, 2=Female)')
        ax1.set_ylabel('Screen Time (hours)')
        
        # Add statistical test
        male_screen = df[df['SC_SEX'] == 1]['SCREENTIME']
        female_screen = df[df['SC_SEX'] == 2]['SCREENTIME']
        stat, pval = stats.ttest_ind(male_screen.dropna(), female_screen.dropna())
        ax1.text(0.5, -0.15, f'T-test p-value: {pval:.3f}', 
                ha='center', transform=ax1.transAxes)
        
        # Sleep Distribution by Sex
        sns.boxplot(data=df, x='SC_SEX', y='HOURSLEEP', ax=ax2)
        ax2.set_title('Sleep Duration Distribution by Sex')
        ax2.set_xlabel('Sex (1=Male, 2=Female)')
        ax2.set_ylabel('Sleep Duration (hours)')
        
        # Add statistical test
        male_sleep = df[df['SC_SEX'] == 1]['HOURSLEEP']
        female_sleep = df[df['SC_SEX'] == 2]['HOURSLEEP']
        stat, pval = stats.ttest_ind(male_sleep.dropna(), female_sleep.dropna())
        ax2.text(0.5, -0.15, f'T-test p-value: {pval:.3f}', 
                ha='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig('visualizations/sex_distributions.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print sex-based statistics
        print("\nScreen Time Statistics by Sex:")
        print(df.groupby('SC_SEX')['SCREENTIME'].describe())
        print("\nSleep Duration Statistics by Sex:")
        print(df.groupby('SC_SEX')['HOURSLEEP'].describe())
        
    # 3. Emotional/Behavioral Correlations
    plt.figure(figsize=(15, 10))
    
    # Combine emotional and behavioral variables
    analysis_vars = EMOTIONAL_VARS + SOCIAL_VARS + ['SCREENTIME', 'HOURSLEEP']
    corr_matrix = df[analysis_vars].corr()
    
    # Update labels for better readability
    labels = {**EMOTIONAL_VARS_LABELS, **SOCIAL_VARS_LABELS}
    labels.update({'SCREENTIME': 'Screen Time', 'HOURSLEEP': 'Sleep Duration'})
    
    corr_matrix.columns = [labels.get(col, col) for col in corr_matrix.columns]
    corr_matrix.index = [labels.get(idx, idx) for idx in corr_matrix.index]
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdYlBu_r',
                center=0,
                fmt='.2f',
                square=True)
    
    plt.title('Correlation Matrix: Screen Time, Emotional, and Behavioral Indicators', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 3. High Screen Time Analysis
    high_screen = df[df['SCREENTIME'] > df['SCREENTIME'].quantile(0.75)]
    
    print("\nAnalysis of High Screen Time Group (Top 25%):")
    print("\nAverage Values:")
    for var in EMOTIONAL_VARS + SOCIAL_VARS + SLEEP_VARS:
        var_label = labels.get(var, var)
        print(f"{var_label}: {high_screen[var].mean():.2f}")

    # Save detailed analysis to Excel
    with pd.ExcelWriter('screen_time_analysis.xlsx') as writer:
        # Overall correlations
        corr_matrix.to_excel(writer, sheet_name='Overall_Correlations')
        
        # High screen time group analysis
        high_screen.describe().to_excel(writer, sheet_name='High_Screen_Time_Group')
        
        # Sex-based analysis (if available)
        if 'SC_SEX' in df.columns:
            # Screen time by sex
            df.pivot_table(
                values='SCREENTIME',
                index='SC_SEX',
                aggfunc=['mean', 'std', 'count']
            ).to_excel(writer, sheet_name='Screen_Time_by_Sex')
            
            # Sleep by sex
            df.pivot_table(
                values='HOURSLEEP',
                index='SC_SEX',
                aggfunc=['mean', 'std', 'count']
            ).to_excel(writer, sheet_name='Sleep_by_Sex')
            
            # Emotional indicators by sex
            df.pivot_table(
                values=EMOTIONAL_VARS,
                index='SC_SEX',
                aggfunc='mean'
            ).to_excel(writer, sheet_name='Emotional_by_Sex')
            
            # Social indicators by sex
            df.pivot_table(
                values=SOCIAL_VARS,
                index='SC_SEX',
                aggfunc='mean'
            ).to_excel(writer, sheet_name='Social_by_Sex')
        
        # Sleep patterns
        df.groupby(pd.qcut(df['SCREENTIME'], q=4))[SLEEP_VARS].mean().to_excel(
            writer, sheet_name='Sleep_by_Screen_Time_Quartile')
        
        # Behavioral patterns
        df.groupby(pd.qcut(df['SCREENTIME'], q=4))[SOCIAL_VARS].mean().to_excel(
            writer, sheet_name='Behavior_by_Screen_Time_Quartile')

    return df

def main():
    """Main analysis pipeline."""
    # Load and preprocess data
    df = load_nsch_data()
    
    # Analyze risk factors
    df_with_risk = analyze_risk_factors(df)
    
    print("Analysis complete. Check 'visualizations' folder for results.")
    print("Detailed analysis saved to 'screen_time_analysis.xlsx'")

if __name__ == "__main__":
    main()
