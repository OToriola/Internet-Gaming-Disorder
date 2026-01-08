import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report, brier_score_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data():
    """
    Placeholder for data loading and preprocessing.
    In actual implementation, this will combine relevant features from both NSCH and IGD datasets.
    """
    # Key features from both datasets:
    # NSCH: Screen time, emotional/behavioral scores, sleep patterns, physical activity
    # IGD: Gaming hours, sleep quality, social factors, IGD diagnostic criteria
    
    # For proposal demonstration, using IGD dataset
    df = pd.read_excel("data/IGD Database.xlsx")
    
    # Convert categorical variables to numeric - STANDARDIZED APPROACH
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
    
    # Select relevant features
    features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 
               'IGD Total', 'Social', 'Escape']
    target = 'IGD Status'
    
    # Convert numeric features
    for col in ['Sleep Quality', 'IGD Total', 'Social', 'Escape']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values in features
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
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    return X, y

def evaluate_models(X, y):
    """Evaluate multiple models and return their performance metrics"""
    # Stratified split to preserve class distribution (important for imbalanced IGD data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Print class distribution to verify stratification
    print("\n=== Class Distribution ===")
    print(f"Overall dataset - IGD Positive: {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")
    print(f"Training set - IGD Positive: {(y_train == 1).sum()} ({(y_train == 1).mean() * 100:.1f}%)")
    print(f"Test set - IGD Positive: {(y_test == 1).sum()} ({(y_test == 1).mean() * 100:.1f}%)")
    print("=" * 45)
    
    # Calculate class weight for imbalance handling
    n_neg = len(y_train[y_train == 0])
    n_pos = len(y_train[y_train == 1])
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\nClass Imbalance Weights:")
    print(f"Negative (N) samples: {n_neg}")
    print(f"Positive (Y) samples: {n_pos}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    print("=" * 45)
    
    # Initialize models with pipelines and class balancing for imbalanced data
    # Scaling is included in pipelines for models that need it (LR, SVM)
    # Tree-based models don't require scaling
    # Class balancing improves recall for minority IGD positive class
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': ('scaled', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
        'Random Forest': ('unscaled', RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)),
        'SVM': ('scaled', SVC(probability=True, class_weight='balanced', random_state=42)),
        'Gradient Boosting': ('unscaled', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
        'XGBoost': ('unscaled', XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=10, random_state=42, verbosity=0)),
        'LightGBM': ('unscaled', LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, verbose=-1))
    }

    # Store results and detailed reports
    results = []
    reports = {}
    confusion_matrices = {}

    for name, (data_type, model) in models.items():
        # Use scaled or unscaled data based on model type
        X_tr = X_train_scaled if data_type == 'scaled' else X_train
        X_te = X_test_scaled if data_type == 'scaled' else X_test
        
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        # Calculate metrics (pos_label not needed since target is already 0/1)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Get probability predictions for AUC metrics
        auc_roc = None
        pr_auc = None
        brier = None
        try:
            y_prob = model.predict_proba(X_te)[:, 1]
            auc_roc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            brier = brier_score_loss(y_test, y_prob)
        except Exception as e:
            pass

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc_roc,
            'PR-AUC': pr_auc,
            'Brier Score': brier
        })
        reports[name] = classification_report(y_test, y_pred, output_dict=True)
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    # Add Deep Learning (MLP) model with fixed architecture
    print("\nTraining Deep Learning (MLP) model...")
    
    model_dl = Sequential([
        Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model_dl.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_dl.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    y_pred_dl = (model_dl.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred_dl)
    precision = precision_score(y_test, y_pred_dl, zero_division=0)
    recall = recall_score(y_test, y_pred_dl, zero_division=0)
    f1 = f1_score(y_test, y_pred_dl, zero_division=0)
    auc_roc_dl = None
    pr_auc_dl = None
    brier_dl = None
    try:
        y_prob_dl = model_dl.predict(X_test_scaled, verbose=0).flatten()
        auc_roc_dl = roc_auc_score(y_test, y_prob_dl)
        pr_auc_dl = average_precision_score(y_test, y_prob_dl)
        brier_dl = brier_score_loss(y_test, y_prob_dl)
    except Exception as e:
        pass
    results.append({
        'Model': 'Deep Learning (MLP, Tuned)',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc_dl,
        'PR-AUC': pr_auc_dl,
        'Brier Score': brier_dl
    })
    reports['Deep Learning (MLP, Tuned)'] = classification_report(y_test, y_pred_dl, output_dict=True)
    confusion_matrices['Deep Learning (MLP, Tuned)'] = confusion_matrix(y_test, y_pred_dl)

    # Return results and detailed reports
    return pd.DataFrame(results), reports, confusion_matrices

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Evaluate models
    results, reports, confusion_matrices = evaluate_models(X, y)

    # Display results
    print("\nModel Evaluation Results:")
    print(results.to_string(index=False))
    results.to_csv('igd_model_evaluation_results.csv', index=False)

    # Save classification reports and confusion matrices to CSV files
    os.makedirs('igd_results', exist_ok=True)
    
    # Save all classification reports to a single CSV
    all_reports = []
    for model_name, report_dict in reports.items():
        report_df = pd.DataFrame(report_dict).transpose()
        report_df['Model'] = model_name
        all_reports.append(report_df)
    
    combined_reports = pd.concat(all_reports, ignore_index=False)
    combined_reports.to_csv('igd_results/igd_classification_reports.csv')
    
    # Save confusion matrices to a single CSV
    with open('igd_results/igd_confusion_matrices.csv', 'w') as f:
        for model_name, cm in confusion_matrices.items():
            f.write(f"\n{model_name}\n")
            cm_df = pd.DataFrame(cm, columns=['Predicted Negative', 'Predicted Positive'], 
                                index=['Actual Negative', 'Actual Positive'])
            cm_df.to_csv(f)
    
    # Print detailed classification reports and confusion matrices to console
    for model_name in reports:
        print(f"\nClassification Report for {model_name}:")
        print(pd.DataFrame(reports[model_name]).transpose())
        print(f"\nConfusion Matrix for {model_name}:")
        print(confusion_matrices[model_name])
    
    print("\n" + "="*50)
    print("All results saved successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
