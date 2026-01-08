import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Set seeds for reproducibility
np.random.seed(42)

# Load and preprocess data
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

# Convert numeric features
for col in ['Sleep Quality', 'IGD Total', 'Social', 'Escape']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

features = ['Weekday Hours', 'Weekend Hours', 'Sleep Quality', 'IGD Total', 'Social', 'Escape']
target = 'IGD Status'

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

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights
n_neg = len(y_train[y_train == 0])
n_pos = len(y_train[y_train == 1])
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

print("\n=== Class Distribution ===")
print(f"Overall dataset - IGD Positive: {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")
print(f"Training set - IGD Positive: {(y_train == 1).sum()} ({(y_train == 1).mean() * 100:.1f}%)")
print(f"Test set - IGD Positive: {(y_test == 1).sum()} ({(y_test == 1).mean() * 100:.1f}%)")
print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
print("=" * 45)

# Test pipelines with updated metrics
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
}

print("\n=== Model Performance with AUC Metrics ===")
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics (simplified - no pos_label)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Get AUC metrics
    auc_roc = None
    pr_auc = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
    except Exception as e:
        pass
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'PR-AUC': pr_auc
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_roc:.4f}" if auc_roc else "  AUC-ROC:   N/A")
    print(f"  PR-AUC:    {pr_auc:.4f}" if pr_auc else "  PR-AUC:    N/A")

# Display results as DataFrame
print("\n=== Summary Table ===")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
