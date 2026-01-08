import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import shap

def load_and_combine_data():
    """Load both NSCH and IGD datasets and combine relevant features"""
    # Create visualization directory if it doesn't exist
    if not os.path.exists('nsch_visualizations'):
        os.makedirs('nsch_visualizations')
    
    # Load datasets
    nsch_df = pd.read_excel("nsch_2023e_topical.xlsx")
    print("\nAvailable columns in NSCH dataset:")
    print(nsch_df.columns.tolist())
    
    # NSCH key features - updated with actual column names
    nsch_features = {
        'screen_time': 'SCREENTIME',  # Screen time
        'emotional_support': 'K7Q70_R',  # Emotional support
        'mental_health': 'A1_MENTHEALTH',  # Mental health status
        'anxiety': 'K2Q31A',  # Anxiety problems
        'depression': 'K2Q32A',  # Depression
        'behavior': 'K2Q33A',  # Behavior problems
        'social_behavior': 'PLAYWELL',  # Social behavior
        'sleep': 'HOURSLEEP',  # Sleep hours
        'physical_activity': 'PHYSACTIV'  # Physical activity
    }
    
    # Select relevant NSCH features
    nsch_selected = nsch_df[list(nsch_features.values())].copy()
    
    # Analyze relationship between screen time and mental health indicators
    print("\nScreen Time Impact Analysis from NSCH Data:")
    
    # Screen time vs Mental Health
    screen_mental = pd.crosstab(nsch_df['SCREENTIME'], nsch_df['A1_MENTHEALTH'])
    print("\nScreen Time vs Mental Health Status:")
    print(screen_mental)
    
    # Create visualization for screen time vs mental health
    plt.figure(figsize=(10, 6))
    screen_mental_pct = screen_mental.div(screen_mental.sum(axis=1), axis=0)
    screen_mental_pct.plot(kind='bar', stacked=True)
    plt.title('Screen Time vs Mental Health Status')
    plt.xlabel('Screen Time Hours')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig('nsch_visualizations/screen_time_mental_health.png')
    plt.close()
    
    # Analyze emotional/social consequences
    emotional_factors = ['K2Q31A', 'K2Q32A', 'K2Q33A', 'PLAYWELL']
    screen_emotional = nsch_df.groupby('SCREENTIME')[emotional_factors].mean()
    print("\nScreen Time vs Emotional/Social Factors (Average Scores):")
    print(screen_emotional)
    
    # Create visualization for emotional factors
    plt.figure(figsize=(12, 6))
    screen_emotional.plot(kind='bar')
    plt.title('Screen Time vs Emotional/Behavioral Factors')
    plt.xlabel('Screen Time Hours')
    plt.ylabel('Average Score')
    plt.legend(['Anxiety', 'Depression', 'Behavior Problems'])
    plt.tight_layout()
    plt.savefig('nsch_visualizations/screen_time_emotional.png')
    plt.close()
    
    # Calculate correlation matrix
    correlation_vars = ['SCREENTIME', 'A1_MENTHEALTH', 'K2Q31A', 'K2Q32A', 'K2Q33A', 
                       'K7Q70_R', 'HOURSLEEP', 'PHYSACTIV', 'PLAYWELL']
    correlation_matrix = nsch_df[correlation_vars].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Screen Time and Health Indicators')
    plt.tight_layout()
    plt.savefig('nsch_visualizations/correlation_heatmap.png')
    plt.close()
    
    # Additional analysis: Social behavior and screen time
    social_screen = pd.crosstab(nsch_df['SCREENTIME'], nsch_df['PLAYWELL'])
    print("\nScreen Time vs Social Behavior:")
    print(social_screen)
    
    return nsch_selected

def analyze_combined_impact():
    """Analyze the combined impact of screen time and gaming on mental health"""
    nsch_data = load_and_combine_data()
    
    # Calculate crosstabs for tables
    screen_mental = pd.crosstab(nsch_data['SCREENTIME'], nsch_data['A1_MENTHEALTH'])
    emotional_factors = ['K2Q31A', 'K2Q32A', 'K2Q33A', 'PLAYWELL']
    screen_emotional = nsch_data.groupby('SCREENTIME')[emotional_factors].mean()
    
    # Print summary statistics
    print("\nSummary Statistics for Screen Time and Mental Health Indicators:")
    print(nsch_data.describe())
    
    # Calculate risk factors based on screen time and mental health indicators
    # Convert screen time to numeric if needed
    nsch_data['SCREENTIME'] = pd.to_numeric(nsch_data['SCREENTIME'], errors='coerce')
    
    high_risk = (nsch_data['SCREENTIME'] > 3) & \
                ((nsch_data['A1_MENTHEALTH'] <= 2) | \
                 (nsch_data['K2Q31A'] == 1) | \
                 (nsch_data['K2Q32A'] == 1))
    
    # Create formatted tables for the results
    print("\n=== Screen Time Impact on Mental Health and Behavior ===")
    print("\nTable 1: Mental Health Status Distribution by Screen Time (Count)")
    print("-" * 60)
    print(f"{'Screen Time':12} {'Excellent':>10} {'Very Good':>10} {'Good':>10} {'Fair':>10} {'Poor':>10}")
    print("-" * 60)
    for screen_time in sorted(screen_mental.index):
        row = screen_mental.loc[screen_time]
        print(f"{screen_time:12.1f} {row[1.0]:10.0f} {row[2.0]:10.0f} {row[3.0]:10.0f} {row[4.0]:10.0f} {row[5.0]:10.0f}")
    print("-" * 60)
    
    print("\nTable 2: Emotional and Social Outcomes by Screen Time")
    print("-" * 70)
    print(f"{'Screen Time':12} {'Anxiety':>12} {'Depression':>12} {'Behavior':>12} {'Social':>12}")
    print("            (lower is worse) (lower is worse) (lower is worse) (lower is worse)")
    print("-" * 70)
    for screen_time in sorted(screen_emotional.index):
        row = screen_emotional.loc[screen_time]
        print(f"{screen_time:12.1f} {row['K2Q31A']:12.3f} {row['K2Q32A']:12.3f} {row['K2Q33A']:12.3f} {row['PLAYWELL']:12.3f}")
    print("-" * 70)
    
    print("\nTable 3: Summary Statistics")
    print("-" * 50)
    metrics = {
        "High Risk Prevalence": f"{(high_risk.mean() * 100):.1f}%",
        "Average Screen Time": f"{nsch_data['SCREENTIME'].mean():.1f} hours",
        "Average Sleep": f"{nsch_data['HOURSLEEP'].mean():.1f} hours",
        "Physical Activity Score": f"{nsch_data['PHYSACTIV'].mean():.1f}/4"
    }
    for metric, value in metrics.items():
        print(f"{metric:25} {value:>20}")
    print("-" * 50)
    
    # Create risk prediction model
    features = ['SCREENTIME', 'A1_MENTHEALTH', 'K2Q31A', 'K2Q32A', 'K2Q33A', 'HOURSLEEP', 'PHYSACTIV']
    X = nsch_data[features].fillna(nsch_data[features].mean())
    y = high_risk.astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Feature importance using SHAP
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('Feature Importance for Risk Prediction')
    plt.tight_layout()
    plt.savefig('nsch_visualizations/feature_importance.png')
    plt.close()

def main():
    print("Analyzing Screen Time and Mental Health Relationships")
    analyze_combined_impact()

if __name__ == "__main__":
    main()
