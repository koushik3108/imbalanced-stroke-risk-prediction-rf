import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportions_ztest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from imblearn.over_sampling import SMOTE

# Load and clean data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Check for nulls
null_vals_col = df.isnull().sum()
print("Null values per column:\n", null_vals_col, "\n")

# Posting Null values per column to excel
null_vals_col.to_excel('Null Values per Column.xlsx')

# Impute missing BMI with median
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Descriptive statistics
desc_stats = df.describe(include='all')
print("Descriptive statistics:\n", desc_stats, "\n")

# Posting descriptive statistics to excel
desc_stats.to_excel('Descriptive Statistics.xlsx')

# # Histograms for numeric columns (separate plots)
numeric_cols = df.select_dtypes(include=np.number).columns.drop('id')
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# # Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# # Count plots for categorical variables
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Drop identifier
df_model = df.drop(columns=['id'], axis=1)

# One-hot encode categorical predictors
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

X = df_model.drop('stroke', axis=1)
y = df_model['stroke']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print class distribution before SMOTE
print(f"\nTotal observations: {len(df)}")
print(f"Training set (80%): {len(X_train)} "
      f"(stroke = {y_train.sum()}, no stroke = {len(y_train) - y_train.sum()})")
print(f"Test set (20%): {len(X_test)} "
      f"(stroke = {y_test.sum()}, no stroke = {len(y_test) - y_test.sum()})")
print(f"\nClass imbalance ratio (before SMOTE): "
      f"{(len(y_train) - y_train.sum()) / y_train.sum():.1f}:1 (no stroke : stroke)")

# ─── SMOTE: Oversample minority class (stroke=1) on training data only ───────
print("\nApplying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set after SMOTE: {len(X_train_smote)} "
      f"(stroke = {y_train_smote.sum()}, no stroke = {len(y_train_smote) - y_train_smote.sum()})")
print(f"Class balance after SMOTE: "
      f"{(len(y_train_smote) - y_train_smote.sum()) / y_train_smote.sum():.1f}:1 (no stroke : stroke)")

# Plot class distribution before and after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['No Stroke', 'Stroke'],
            [(y_train == 0).sum(), (y_train == 1).sum()],
            color=['steelblue', 'tomato'])
axes[0].set_title('Training Set — Before SMOTE')
axes[0].set_ylabel('Count')
for ax, counts in [(axes[0], y_train), (axes[1], y_train_smote)]:
    vals = [(counts == 0).sum(), (counts == 1).sum()]
    for i, v in enumerate(vals):
        ax.text(i, v + 10, str(v), ha='center', fontweight='bold')

axes[1].bar(['No Stroke', 'Stroke'],
            [(y_train_smote == 0).sum(), (y_train_smote == 1).sum()],
            color=['steelblue', 'tomato'])
axes[1].set_title('Training Set — After SMOTE')
axes[1].set_ylabel('Count')
plt.suptitle('Class Distribution Before vs After SMOTE', fontsize=13)
plt.tight_layout()
plt.show()
# ─────────────────────────────────────────────────────────────────────────────

# ─── Model 1: Without SMOTE ──────────────────────────────────────────────────
print("\n" + "="*55)
print("  Model 1: Random Forest WITHOUT SMOTE")
print("="*55)
rf_no_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_no_smote.fit(X_train, y_train)
y_pred_no_smote = rf_no_smote.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred_no_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_no_smote))
# ─────────────────────────────────────────────────────────────────────────────

# ─── Model 2: With SMOTE ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Model 2: Random Forest WITH SMOTE")
print("="*55)
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_smote))
# ─────────────────────────────────────────────────────────────────────────────

# Z-test on SMOTE model
n_correct = (y_test == y_pred_smote).sum()
n_total   = len(y_test)

stat, pval = proportions_ztest(
    count=n_correct,
    nobs=n_total,
    value=0.5,
    alternative='larger'
)

print(f"\nZ-test (SMOTE model)")
print(f"Test statistic (z) = {stat:.2f}")
print(f"One-sided p-value   = {pval:.3e}")

# ─── Side-by-side confusion matrices ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, y_pred, title in [
    (axes[0], y_pred_no_smote, 'Without SMOTE'),
    (axes[1], y_pred_smote,    'With SMOTE'),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion Matrix — {title}')
plt.tight_layout()
plt.show()
# ─────────────────────────────────────────────────────────────────────────────

# Feature importances (SMOTE model)
importances = pd.Series(rf_smote.feature_importances_, index=X.columns)
top_imp = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_imp.values, y=top_imp.index)
plt.title('Top 10 Feature Importances (SMOTE Model)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Posting Top 10 Predictors to excel
top_imp.to_excel('Top 10 Predictors.xlsx')
