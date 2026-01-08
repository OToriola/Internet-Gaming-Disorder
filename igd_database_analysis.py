import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set large font sizes for visualizations
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 28,
    "axes.titlesize": 44,
    "axes.labelsize": 36,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 28,
    "figure.titlesize": 46
})

def load_and_examine_igd_data():
    """Load and examine the IGD dataset"""
    print("Loading IGD Database...")
    df = pd.read_excel("data/IGD Database.xlsx")
    
    # Basic dataset information
    print("\nDataset Overview:")
    print(f"Number of records: {len(df)}")
    print(f"Number of variables: {len(df.columns)}")
    print("\nColumns in the dataset:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    # Create visualizations directory if it doesn't exist
    if not os.path.exists('igd_visualizations'):
        os.makedirs('igd_visualizations')
    
    return df

def analyze_igd_patterns(df):
    """Analyze patterns in IGD diagnosis and related factors"""
    print("\nAnalyzing IGD patterns...")
    
    # 1. IGD prevalence analysis
    print("\nIGD Status Distribution:")
    igd_status = df['IGD Status'].value_counts()
    print(igd_status)
    
    # Create bar plot for IGD Status
    plt.figure(figsize=(20, 14))
    ax = sns.barplot(x=igd_status.index, y=igd_status.values, width=0.6)
    plt.title('Distribution of IGD Status', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('IGD Status', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fontsize=11, fontweight='bold', padding=10)
    plt.tight_layout()
    plt.savefig('igd_visualizations/igd_status_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gaming time analysis
    print("\nGaming Time Statistics:")
    print("\nWeekday Gaming Hours:")
    print(df['Weekday Hours'].describe())
    print("\nWeekend Gaming Hours:")
    print(df['Weekend Hours'].describe())
    
    # 3. IGD Score Distribution
    plt.figure(figsize=(20, 14))
    sns.histplot(data=df, x='IGD Total', bins=20, edgecolor='black', linewidth=2)
    plt.title('Distribution of IGD Total Scores', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('IGD Total Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('igd_visualizations/igd_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Display IGD score statistics
    print("\nIGD Total Score Statistics:")
    print(df['IGD Total'].describe())
    
    # Analyze categorical relationships
    print("\nIGD Status by Gaming Time Categories:")
    weekday_igd = pd.crosstab(df['Weekday Hours'], df['IGD Status'])
    print("\nWeekday Gaming Hours vs IGD Status:")
    print(weekday_igd)
    
    weekend_igd = pd.crosstab(df['Weekend Hours'], df['IGD Status'])
    print("\nWeekend Gaming Hours vs IGD Status:")
    print(weekend_igd)
    
    # Create stacked bar plots
    plt.figure(figsize=(28, 12))
    
    plt.subplot(1, 2, 1)
    weekday_igd_pct = weekday_igd.div(weekday_igd.sum(axis=1), axis=0)
    weekday_igd_pct.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('IGD Status by Weekday Gaming Hours', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Gaming Hours', fontsize=12, fontweight='bold')
    plt.ylabel('Proportion', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, title='IGD Status', title_fontsize=10)
    
    plt.subplot(1, 2, 2)
    weekend_igd_pct = weekend_igd.div(weekend_igd.sum(axis=1), axis=0)
    weekend_igd_pct.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('IGD Status by Weekend Gaming Hours', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Gaming Hours', fontsize=12, fontweight='bold')
    plt.ylabel('Proportion', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10, title='IGD Status', title_fontsize=10)
    
    plt.tight_layout()
    plt.savefig('igd_visualizations/gaming_hours_vs_igd.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Gender differences
    print("\nIGD Total Score by Gender:")
    gender_stats = df.groupby('Sex')['IGD Total'].describe()
    print(gender_stats)
    
    # Box plot for gender differences
    plt.figure(figsize=(16, 12))
    sns.boxplot(x='Sex', y='IGD Total', data=df, linewidth=3)
    plt.title('IGD Total Score Distribution by Gender', fontsize=48, fontweight='bold', pad=30)
    plt.xlabel('Gender', fontsize=40, fontweight='bold', labelpad=15)
    plt.ylabel('IGD Total Score', fontsize=40, fontweight='bold', labelpad=15)
    plt.xticks(fontsize=36, fontweight='bold')
    plt.yticks(fontsize=36)
    plt.tight_layout()
    plt.savefig('igd_visualizations/gender_differences.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Sleep analysis
    print("\nSleep Quality Distribution:")
    sleep_quality = df['Sleep Quality'].value_counts()
    print(sleep_quality)
    
    # Calculate average IGD score by sleep quality
    sleep_igd = df.groupby('Sleep Quality')['IGD Total'].mean()
    print("\nAverage IGD Score by Sleep Quality:")
    print(sleep_igd)

def main():
    # Load and examine the IGD dataset
    igd_df = load_and_examine_igd_data()
    
    # Analyze IGD patterns
    analyze_igd_patterns(igd_df)
    
if __name__ == "__main__":
    main()
