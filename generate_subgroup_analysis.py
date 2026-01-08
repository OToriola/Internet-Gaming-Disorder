"""
Generate Subgroup Analysis for IGD Models
Evaluates model performance across demographic subgroups (sex and age)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import random
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("SUBGROUP ANALYSIS - SEX AND AGE GROUPS")
print("="*80)

# Load data
print("\n1. Loading IGD data...")
data = pd.read_excel('data/IGD Database.xlsx')
print(f"   Loaded {len(data)} samples")

# Convert hour columns to numeric
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

data['Weekday Hours'] = data['Weekday Hours'].apply(convert_hours_to_numeric)
data['Weekend Hours'] = data['Weekend Hours'].apply(convert_hours_to_numeric)

# Convert other columns to numeric
for col in ['Sleep Quality', 'IGD Total', 'Social', 'Escape']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Define features and target
features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 
           'IGD Total', 'Social', 'Escape']
X = data[features].dropna()

# Convert target to binary
if data['IGD Status'].dtype == 'object':
    y = (data.loc[X.index, 'IGD Status'].astype(str).str.strip() == 'Y').astype(int)
else:
    y = data.loc[X.index, 'IGD Status'].astype(int)

# Remove any NaN in target
mask = ~y.isna()
X = X[mask]
y = y[mask]

# Get demographic columns
sex = data.loc[X.index, 'Sex'].reset_index(drop=True)
age = data.loc[X.index, 'Age'].reset_index(drop=True)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"   After removing missing values: {len(X)} samples")
print(f"   IGD Positive: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test, sex_train, sex_test, age_train, age_test = train_test_split(
    X, y, sex, age, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n2. Test set demographics:")
print(f"   Total: {len(X_test)} samples")
print(f"   Males: {(sex_test == 'M').sum()}, IGD+: {y_test[sex_test == 'M'].sum()}")
print(f"   Females: {(sex_test == 'F').sum()}, IGD+: {y_test[sex_test == 'F'].sum()}")

print("\n3. Training models and evaluating by subgroup...")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, 
                                          class_weight='balanced', n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, 
                           scale_pos_weight=10, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, 
                              class_weight='balanced', verbose=-1),
}

# Store results
results = []

# Train each model
for name, model in models.items():
    print(f"   [{list(models.keys()).index(name)+1}/7] {name}...")
    
    # Train model
    if name in ['Logistic Regression', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
    
    # Evaluate on male subgroup
    male_mask = sex_test == 'M'
    if male_mask.sum() > 0:
        y_true_male = y_test[male_mask]
        y_pred_male = y_pred_test[male_mask]
        
        acc_male = accuracy_score(y_true_male, y_pred_male)
        prec_male = precision_score(y_true_male, y_pred_male, zero_division=0)
        rec_male = recall_score(y_true_male, y_pred_male, zero_division=0)
        f1_male = f1_score(y_true_male, y_pred_male, zero_division=0)
        
        results.append({
            'Model': name,
            'Sex': 'Male',
            'N': male_mask.sum(),
            'IGD+': y_true_male.sum(),
            'Accuracy': acc_male,
            'Precision': prec_male,
            'Recall': rec_male,
            'F1': f1_male
        })
    
    # Evaluate on female subgroup
    female_mask = sex_test == 'F'
    if female_mask.sum() > 0:
        y_true_female = y_test[female_mask]
        y_pred_female = y_pred_test[female_mask]
        
        acc_female = accuracy_score(y_true_female, y_pred_female)
        prec_female = precision_score(y_true_female, y_pred_female, zero_division=0)
        rec_female = recall_score(y_true_female, y_pred_female, zero_division=0)
        f1_female = f1_score(y_true_female, y_pred_female, zero_division=0)
        
        results.append({
            'Model': name,
            'Sex': 'Female',
            'N': female_mask.sum(),
            'IGD+': y_true_female.sum(),
            'Accuracy': acc_female,
            'Precision': prec_female,
            'Recall': rec_female,
            'F1': f1_female
        })

# Deep Learning (MLP)
print("   [7/7] Deep Learning (MLP)...")
mlp = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
mlp.fit(X_train_scaled, y_train, epochs=100, batch_size=16, 
        callbacks=[early_stop], verbose=0)
y_pred_mlp = (mlp.predict(X_test_scaled, verbose=0).flatten() > 0.5).astype(int)

# Male subgroup
male_mask = sex_test == 'M'
if male_mask.sum() > 0:
    y_true_male = y_test[male_mask]
    y_pred_male = y_pred_mlp[male_mask]
    
    acc_male = accuracy_score(y_true_male, y_pred_male)
    prec_male = precision_score(y_true_male, y_pred_male, zero_division=0)
    rec_male = recall_score(y_true_male, y_pred_male, zero_division=0)
    f1_male = f1_score(y_true_male, y_pred_male, zero_division=0)
    
    results.append({
        'Model': 'Deep Learning (MLP)',
        'Sex': 'Male',
        'N': male_mask.sum(),
        'IGD+': y_true_male.sum(),
        'Accuracy': acc_male,
        'Precision': prec_male,
        'Recall': rec_male,
        'F1': f1_male
    })

# Female subgroup
female_mask = sex_test == 'F'
if female_mask.sum() > 0:
    y_true_female = y_test[female_mask]
    y_pred_female = y_pred_mlp[female_mask]
    
    acc_female = accuracy_score(y_true_female, y_pred_female)
    prec_female = precision_score(y_true_female, y_pred_female, zero_division=0)
    rec_female = recall_score(y_true_female, y_pred_female, zero_division=0)
    f1_female = f1_score(y_true_female, y_pred_female, zero_division=0)
    
    results.append({
        'Model': 'Deep Learning (MLP)',
        'Sex': 'Female',
        'N': female_mask.sum(),
        'IGD+': y_true_female.sum(),
        'Accuracy': acc_female,
        'Precision': prec_female,
        'Recall': rec_female,
        'F1': f1_female
    })

# Create DataFrame
df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("TABLE 11: MODEL PERFORMANCE ACROSS SEX SUBGROUPS")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)

# Save results
df_results.to_csv('visualizations/subgroup_analysis_sex.csv', index=False)
print("\n✓ Results saved to: visualizations/subgroup_analysis_sex.csv")
print("="*80)
