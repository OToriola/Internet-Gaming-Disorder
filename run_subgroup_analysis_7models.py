#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UPDATED Subgroup Analysis - All 7 Models
=========================================

This script runs subgroup analysis for your dissertation.
NOW INCLUDES ALL 7 MODELS:
- Logistic Regression
- Random Forest
- SVM
- Gradient Boosting
- XGBoost
- LightGBM
- Deep Learning MLP

USAGE:
  python run_subgroup_analysis_7models.py
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

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
                print(f"[✓] Found IGD data at: {path}")
                break
        
        if df is None:
            print("[ERROR] IGD Database.xlsx not found")
            return None, None, None
        
        # Mapping for hour ranges
        def convert_hours(val):
            mapping = {
                '1 or less': 0.5, '1 to 2': 1.5, '2 to 3': 2.5, '3 to 4': 3.5,
                '4 to 5': 4.5, '5 to 6': 5.5, '6 to 7': 6.5, '7 to 8': 7.5,
                '8 to 9': 8.5, '9 to 10': 9.5, '10 or more': 10.5,
            }
            if pd.isna(val):
                return np.nan
            return mapping.get(str(val).strip(), np.nan)
        
        df['Weekday Hours'] = df['Weekday Hours'].apply(convert_hours)
        df['Weekend Hours'] = df['Weekend Hours'].apply(convert_hours)
        
        features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 
                   'IGD Total', 'Social', 'Escape']
        target = 'IGD Status'
        
        # Remove NaN
        df = df.dropna(subset=features + [target])
        df[target] = (df[target].astype(str).str.strip() == 'Y').astype(int)
        
        X = df[features].reset_index(drop=True)
        y = df[target].reset_index(drop=True)
        df_full = df.reset_index(drop=True)
        
        print(f"[✓] Loaded IGD dataset: {len(df)} samples, {y.sum()} positive\n")
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
    
    # Sex
    if 'Sex' in df_full.columns:
        demographics['sex'] = df_full['Sex'].map({'M': 'Male', 'F': 'Female'})
    else:
        demographics['sex'] = 'Unknown'
    
    # Age groups
    if 'Age' in df_full.columns:
        demographics['age_group'] = pd.cut(
            df_full['Age'],
            bins=[0, 17, 100],
            labels=['15-17 years', '18+ years']
        )
    else:
        demographics['age_group'] = 'Unknown'
    
    return demographics


# ============================================================================
# TRAIN ALL 7 MODELS
# ============================================================================

def train_all_7_models(X, y):
    """Train all 7 models for subgroup analysis"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    import xgboost as xgb
    import lightgbm as lgb
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training all 7 models...")
    models = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, 
                               class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 3. SVM
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    models['SVM'] = svm
    
    # 4. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    
    # 5. XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, 
                                  scale_pos_weight=10, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # 6. LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, 
                                   class_weight='balanced', verbose=-1, random_state=42)
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    
    # 7. Deep Learning MLP
    try:
        mlp = Sequential([
            Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        mlp.fit(X_train_scaled, y_train, epochs=100, batch_size=16, 
               validation_split=0.2, callbacks=[early_stop], verbose=0)
        models['Deep Learning (MLP)'] = mlp
        print("  ✓ All 7 models trained successfully\n")
    except Exception as e:
        print(f"  ⚠ MLP training failed: {str(e)[:50]}")
        print("  Using 6 models instead\n")
    
    return models, X_test, y_test, X_test_scaled


# ============================================================================
# EVALUATE BY SUBGROUP
# ============================================================================

def evaluate_subgroups(models, X, X_scaled, y, demographics, feature_names):
    """Evaluate each model in subgroups"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    results = []
    
    for subgroup_col in ['sex', 'age_group']:
        print(f"\n{'='*80}")
        print(f"SUBGROUP ANALYSIS BY {subgroup_col.upper()}")
        print(f"{'='*80}\n")
        
        subgroups = demographics[subgroup_col].unique()
        subgroups = [s for s in subgroups if s != 'Unknown']
        
        for subgroup in subgroups:
            mask = demographics[subgroup_col] == subgroup
            X_sub = X[mask]
            X_sub_scaled = X_scaled[mask]
            y_sub = y[mask]
            
            n_total = len(y_sub)
            n_positive = y_sub.sum()
            
            print(f"{subgroup} (n={n_total}, positive={n_positive}):")
            print("-" * 80)
            
            row_data = {
                'Subgroup': subgroup_col.capitalize(),
                'Subgroup Value': subgroup,
                'N Total': n_total,
                'N Positive': n_positive
            }
            
            # Evaluate each model
            for model_name, model in models.items():
                try:
                    # Get predictions
                    if model_name in ['Logistic Regression', 'SVM']:
                        y_pred_proba = model.predict_proba(X_sub_scaled)[:, 1]
                    elif model_name == 'Deep Learning (MLP)':
                        try:
                            y_pred_proba = model.predict(X_sub_scaled, verbose=0).flatten()
                        except:
                            continue
                    else:
                        y_pred_proba = model.predict_proba(X_sub)[:, 1]
                    
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # Metrics
                    acc = accuracy_score(y_sub, y_pred)
                    prec = precision_score(y_sub, y_pred, zero_division=0)
                    rec = recall_score(y_sub, y_pred, zero_division=0)
                    f1 = f1_score(y_sub, y_pred, zero_division=0)
                    
                    # Store results
                    row_data[f'{model_name} Acc'] = acc
                    row_data[f'{model_name} Prec'] = prec
                    row_data[f'{model_name} Rec'] = rec
                    row_data[f'{model_name} F1'] = f1
                    
                    print(f"  {model_name:20s} | Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")
                    
                except Exception as e:
                    print(f"  {model_name:20s} | ERROR: {str(e)[:30]}")
            
            results.append(row_data)
            print()
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("SUBGROUP ANALYSIS - ALL 7 MODELS")
    print("="*80 + "\n")
    
    # Load data
    X, y, df_full = load_igd_data()
    if X is None:
        print("[ERROR] Could not load data")
        sys.exit(1)
    
    # Get demographics
    demographics = create_demographics(df_full)
    
    # Train all 7 models
    models, X_test, y_test, X_test_scaled = train_all_7_models(X, y)
    
    # Evaluate by subgroup
    feature_names = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 'IGD Total', 'Social', 'Escape']
    results = evaluate_subgroups(models, X_test, X_test_scaled, y_test, 
                                demographics.loc[X_test.index], feature_names)
    
    # Save results
    results.to_csv('subgroup_analysis_7models_results.csv', index=False)
    print("\n" + "="*80)
    print("✓ Results saved to: subgroup_analysis_7models_results.csv")
    print("="*80)
    
    return results


if __name__ == '__main__':
    main()
