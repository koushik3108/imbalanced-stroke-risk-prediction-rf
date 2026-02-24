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

# Print counts
print(f"Total observations: {len(df)}")
print(f"Training set (80%): {len(X_train)} "
      f"(stroke = {y_train.sum()}, no stroke = {len(y_train) - y_train.sum()})")
print(f"Test set (20%): {len(X_test)} "
      f"(stroke = {y_test.sum()}, no stroke = {len(y_test) - y_test.sum()})")

# Random Forest modeling
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

n_correct = (y_test == y_pred).sum()  # number of correct predictions
n_total   = len(y_test)               # total test cases

# perform one‐sided z–test for a single proportion
stat, pval = proportions_ztest(
    count=n_correct,
    nobs=n_total,
    value=0.5,            # null hypothesis proportion
    alternative='larger'  # one‐sided test: p > 0.5
)

print(f"Test statistic (z) = {stat:.2f}")
print(f"One‐sided p‐value   = {pval:.3e}")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Stroke','Stroke'],
            yticklabels=['No Stroke','Stroke'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# # Feature importances plot
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_imp = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_imp.values, y=top_imp.index)
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Posting Top 10 Predictors to excel
top_imp.to_excel('Top 10 Predictors.xlsx')
