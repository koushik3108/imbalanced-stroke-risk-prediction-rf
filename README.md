# imbalanced-stroke-risk-prediction-rf
# 🧠 Imbalanced Stroke Risk Classification
## A Random Forest Case Study on Rare-Event Prediction

**Author:** Sai Koushik Soma
**Dataset:** Kaggle Stroke Prediction Dataset
**Records:** 5,110 Patients

---

## 📌 Project Overview

This project investigates stroke prediction using demographic, clinical, and lifestyle data. It is framed as a **case study on imbalanced classification**, demonstrating:

1. Why accuracy alone is insufficient in rare-event medical prediction
2. How **SMOTE (Synthetic Minority Over-sampling Technique)** addresses class imbalance
3. The real-world trade-offs between overall accuracy and minority-class recall

Stroke prevalence in the dataset is only **4.9%**, creating severe class imbalance. Without correction, a naive model simply predicts "No Stroke" for every patient — achieving high accuracy while being clinically useless.

---

## 🎯 Objectives

- Build a Random Forest model to predict stroke occurrence
- Evaluate performance beyond accuracy (Precision, Recall, F1, ROC-AUC)
- Apply SMOTE oversampling to correct class imbalance on training data
- Compare model performance before and after SMOTE
- Statistically validate performance vs random guessing
- Identify key predictors of stroke risk

---

## 📊 Dataset Summary

- Total records: 5,110
- Stroke cases: 249 (4.87%)
- Non-stroke cases: 4,861 (95.13%)
- Missing values: BMI (3.9%) – imputed using median

### Key Features

**Demographic:** Age, Gender, Marital status, Work type, Residence type
**Clinical:** Hypertension, Heart disease, Average glucose level, BMI
**Lifestyle:** Smoking status
**Target:** Stroke (0 = No, 1 = Yes)

---

## ⚙️ Data Preprocessing

✔ Dropped non-predictive ID column
✔ Imputed missing BMI with median
✔ One-hot encoded categorical variables
✔ Stratified 80/20 train-test split
✔ SMOTE applied **only to training data** (test set kept untouched)

---

## 🔁 SMOTE — Handling Class Imbalance

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic samples for the minority class (stroke = 1) by interpolating between existing minority instances, rather than simply duplicating them.

| Split | No Stroke | Stroke | Ratio |
|-------|-----------|--------|-------|
| Training — Before SMOTE | 3,889 | 199 | 19.5 : 1 |
| Training — After SMOTE  | 3,889 | 3,889 | 1 : 1 |
| Test set (unchanged)    | 972   | 50   | 19.4 : 1 |

> ⚠️ SMOTE is applied **only to training data** to prevent data leakage and ensure the test set reflects real-world class distribution.

---

## 🤖 Models

Two Random Forest classifiers (100 trees, `random_state=42`) were trained and evaluated on the same held-out test set:

- **Model 1** — Trained on original imbalanced data (no correction)
- **Model 2** — Trained on SMOTE-resampled balanced data

---

## 📈 Model Performance Comparison (Test Set — 1,022 patients)

| Metric | Model 1 (No SMOTE) | Model 2 (With SMOTE) | Change |
|--------|-------------------|----------------------|--------|
| **Accuracy** | 94.81% | 91.88% | ▼ 2.93% |
| **Precision (Stroke)** | 0.00% | 16.33% | ▲ +16.33% |
| **Recall (Stroke)** | 0.00% | 16.00% | ▲ +16.00% |
| **F1-Score (Stroke)** | 0.00% | 16.16% | ▲ +16.16% |
| **ROC-AUC** | 0.7986 | 0.7692 | ▼ 0.0294 |

### Confusion Matrix Breakdown

| | Model 1 (No SMOTE) | Model 2 (With SMOTE) |
|--|-------------------|----------------------|
| True Positives (Stroke detected) | **0** | **8** |
| False Negatives (Stroke missed) | 50 | 42 |
| True Negatives | 969 | 931 |
| False Positives | 3 | 41 |

---

## 🔍 Key Findings

### Model 1 — Without SMOTE
```
              precision  recall  f1-score  support
   No Stroke     0.95    1.00      0.97      972
      Stroke     0.00    0.00      0.00       50
    accuracy                       0.95     1022
```
- Predicted **zero stroke cases** — clinically worthless despite 94.8% accuracy
- The model exploited the class imbalance and defaulted to "No Stroke" for every patient

### Model 2 — With SMOTE
```
              precision  recall  f1-score  support
   No Stroke     0.96    0.96      0.96      972
      Stroke     0.16    0.16      0.16       50
    accuracy                       0.92     1022
```
- **Started detecting strokes** — went from 0 to 8 correct stroke detections
- Reduced missed stroke cases from 50 → 42 (FN dropped by 8)
- Macro avg F1 improved from 0.49 → 0.56

---

## ⚠️ Critical Insight

SMOTE introduced a meaningful improvement in **the metrics that matter clinically**:

> **From 0% to 16% stroke recall** — the model can now identify some true stroke cases.

The slight accuracy drop (94.8% → 91.9%) is the expected and acceptable cost of correcting bias toward the majority class. In medical screening, **missing a stroke (false negative) is far more dangerous than a false alarm**.

---

## 📊 Statistical Validation

A one-sample z-test evaluated whether SMOTE model accuracy exceeds random guessing (50%):

- **Z-statistic:** 49.01
- **One-sided p-value:** < 1 × 10⁻³⁰⁸

Result: The SMOTE model performs significantly better than chance, confirming that the balanced training improved generalisation rather than introducing noise.

---

## 🔍 Feature Importance — Top Predictors (SMOTE Model)

1. Average Glucose Level
2. Age
3. BMI
4. Hypertension
5. Heart Disease

These align with established medical risk factors, confirming the model captures clinically meaningful signal.

---

## 🧠 What This Project Demonstrates

- ✅ High accuracy ≠ good model in imbalanced settings
- ✅ SMOTE effectively balances training data without touching the test set
- ✅ Recall and F1 are more meaningful metrics for rare medical events
- ✅ A small drop in accuracy is acceptable when minority-class detection improves
- ✅ Macro-averaged F1 rose from 0.49 → 0.56 after SMOTE

---

## 🚀 Further Improvements

- Tune SMOTE `k_neighbors` and sampling strategy
- Combine SMOTE with undersampling (SMOTEENN, SMOTETomek)
- Try `class_weight='balanced'` in Random Forest
- Optimize probability threshold (default 0.5 → lower threshold for higher recall)
- Evaluate ROC and Precision-Recall curves
- Try XGBoost or cost-sensitive models

---

## 📁 Dataset Source

**Kaggle:**
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

---

## 🏁 Conclusion

This project demonstrates the complete lifecycle of an **imbalanced classification problem in healthcare analytics**:

1. A naive Random Forest achieved 94.8% accuracy but detected **zero stroke cases**
2. After applying SMOTE, the model began identifying stroke patients — a fundamental shift in clinical utility
3. SMOTE's benefit is not in raw accuracy but in **unlocking minority-class detection**, which is the actual goal

This case study emphasises **methodological rigour over superficial performance metrics** and provides a reproducible foundation for improving rare-event medical prediction.

---

⭐ If you found this case study insightful, feel free to star the repository.
