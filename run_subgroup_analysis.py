#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Subgroup Analysis - Executable Script
======================================

This script runs subgroup analysis for your dissertation.

USAGE:
  1. Customize the section below: "CUSTOMIZE FOR YOUR DATA"
  2. Run: python run_subgroup_analysis.py
"""

import pandas as pd
import numpy as np
import os
import sys

# ============================================================================
# CUSTOMIZE FOR YOUR DATA - CHANGE THESE VALUES
# ============================================================================

DATASET_TYPE = 'IGD'  # Options: 'NSCH', 'IGD'
SEX_COLUMN = 'Sex'  # Column name for sex/gender
AGE_COLUMN = 'Age'     # Column name for age
SEX_MAPPING = {'M': 'Male', 'F': 'Female'}  # Map column values to labels


# ============================================================================
# LOAD DATA
# ============================================================================

def load_igd_data():
    """Load IGD dataset"""
    try:
        # Try possible paths
        paths = [
            "IGD Database.xlsx",
            "IGD_Project/data/IGD Database.xlsx",
            "data/IGD Database.xlsx",
            "../IGD_Project/data/IGD Database.xlsx"
        ]
        
        df = None
        for path in paths:
            if os.path.exists(path):
                df = pd.read_excel(path)
                print(f"[OK] Found IGD data at: {path}")
                break
        
        if df is None:
            print("[ERROR] IGD Database.xlsx not found")
            return None, None, None
        
        # Mapping
        weekday_map = {'1 or less': 1, '2 to 3': 2.5, '4 to 5': 4.5, 
                      '6 to 7': 6.5, '8 to 10': 9}
        weekend_map = {'1 or less': 1, '2 to 3': 2.5, '4 to 5': 4.5, 
                      '6 to 7': 6.5, '8 to 10': 9, '11 or more': 11}
        
        df['Weekday Hours'] = df['Weekday Hours'].map(weekday_map)
        df['Weekend Hours'] = df['Weekend Hours'].map(weekend_map)
        
        features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 
                   'IGD Total', 'Social', 'Escape']
        target = 'IGD Status'
        
        df = df.dropna(subset=features + [target])
        df[target] = (df[target] == 'Y').astype(int)
        
        X = df[features].reset_index(drop=True)
        y = df[target].reset_index(drop=True)
        df_full = df.reset_index(drop=True)
        
        print(f"[OK] Loaded IGD dataset: {len(df)} samples")
        return X, y, df_full
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None, None, None


# ============================================================================
# CREATE DEMOGRAPHICS
# ============================================================================

def create_demographics(df_full):
    """Extract demographic data"""
    demographics = pd.DataFrame(index=range(len(df_full)))
    
    # Sex - use the correct column name and mapping
    if 'Sex' in df_full.columns:
        demographics['sex'] = df_full['Sex'].map({'M': 'Male', 'F': 'Female'})
    else:
        demographics['sex'] = 'Unknown'
    
    # Age
    if 'Age' in df_full.columns:
        demographics['age_group'] = pd.cut(
            df_full['Age'],
            bins=[0, 15, 17, 20],
            labels=['<15 years', '15-17 years', '18+ years']
        )
    else:
        demographics['age_group'] = 'Unknown'
    
    return demographics


# ============================================================================
# RUN ANALYSIS
# ============================================================================

def run_analysis(X, y, df_full):
    """Run subgroup analysis"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    try:
        from subgroup_analysis import SubgroupAnalysis
    except ImportError as e:
        print(f"[ERROR] Cannot import subgroup_analysis: {str(e)}")
        print("[INFO] Make sure subgroup_analysis.py is in the same directory")
        return
    
    print("\n" + "="*80)
    print("RUNNING SUBGROUP ANALYSIS")
    print("="*80)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Positive in test: {(y_test == 1).sum()}")
    
    # Demographics - must match y_test index
    df_test = df_full.loc[X_test.index].reset_index(drop=True)
    demographics = create_demographics(df_test)
    demographics.index = y_test.index  # Align indices
    
    # Train models
    print("\nTraining models...")
    y_pred_dict = {}
    y_prob_dict = {}
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_dict[name] = model.predict(X_test)
        try:
            y_prob_dict[name] = model.predict_proba(X_test)[:, 1]
        except:
            y_prob_dict[name] = None
        
        acc = accuracy_score(y_test, y_pred_dict[name])
        print(f"  [OK] {name}: Accuracy = {acc:.4f}")
    
    # Analyze
    print("\nRunning subgroup analysis...")
    analyzer = SubgroupAnalysis(
        X_test=X_test,
        y_test=y_test,
        y_pred_dict=y_pred_dict,
        y_prob_dict=y_prob_dict,
        demographic_data=demographics
    )
    
    results = analyzer.run_full_analysis()
    
    # Tables
    print("\nGenerating tables...")
    tables = analyzer.generate_subgroup_table(output_format='markdown')
    
    # Visualizations
    print("Generating visualizations...")
    analyzer.visualize_subgroup_performance(save_dir='subgroup_visualizations')
    
    print("\n" + "="*80)
    print("[OK] ANALYSIS COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print("  - subgroup_visualizations/ folder with 4 PNG files")
    print("  - Results displayed above")
    print("\nNext: Copy results into your dissertation")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SUBGROUP ANALYSIS")
    print("="*80 + "\n")
    
    X, y, df_full = load_igd_data()
    
    if X is None or y is None:
        print("\n[ERROR] Could not load data")
        print("Check: Data file exists in the right location")
        sys.exit(1)
    
    run_analysis(X, y, df_full)
