# imbalanced-stroke-risk-prediction-rf
# 🧠 Imbalanced Stroke Risk Classification  
## A Random Forest Case Study on Rare-Event Prediction  

**Author:** Sai Koushik Soma  
**Course:** CSDA 5320 – Analytics Applications Using Python  
**Dataset:** Kaggle Stroke Prediction Dataset  
**Records:** 5,110 Patients  

---

## 📌 Project Overview

This project investigates stroke prediction using demographic, clinical, and lifestyle data. While the Random Forest classifier achieved **94.8% accuracy**, deeper analysis revealed a critical issue:

> The model failed to detect any true stroke cases (0% sensitivity).

Rather than presenting this as a success story, this project is framed as a **case study on imbalanced classification**, demonstrating why accuracy alone is insufficient in rare-event medical prediction.

Stroke prevalence in the dataset is only **4.9%**, creating severe class imbalance.

---

## 🎯 Objectives

- Build a Random Forest model to predict stroke occurrence
- Evaluate performance beyond accuracy
- Statistically validate performance vs random guessing
- Analyze why rare-event detection failed
- Identify key predictors of stroke risk

---

## 📊 Dataset Summary

- Total records: 5,110
- Stroke cases: 249 (4.87%)
- Non-stroke cases: 4,861 (95.13%)
- Missing values: BMI (3.9%) – imputed using median

### Key Features

**Demographic**
- Age
- Gender
- Marital status
- Work type
- Residence type

**Clinical**
- Hypertension
- Heart disease
- Average glucose level
- BMI

**Lifestyle**
- Smoking status

**Target**
- Stroke (0 = No, 1 = Yes)

---

## ⚙️ Data Preprocessing

✔ Dropped non-predictive ID column  
✔ Imputed missing BMI with median  
✔ One-hot encoded categorical variables  
✔ Stratified 80/20 train-test split  
✔ Preserved class imbalance in test set  

---

## 🤖 Model Used

### Random Forest Classifier
- 100 trees
- Default hyperparameters
- No feature scaling required
- Handles mixed data types
- Robust to nonlinear relationships

---

## 📈 Model Performance (Test Set)

| Metric | Value |
|--------|--------|
| Accuracy | 94.8% |
| Sensitivity (Recall – Stroke) | 0.00 |
| Specificity | 1.00 |
| True Positives | 0 |
| False Negatives | 50 |
| True Negatives | 972 |
| False Positives | 0 |

---

## ⚠️ Critical Insight

Although accuracy appears excellent (94.8%), the model:

- Failed to detect any stroke cases
- Defaulted to predicting all observations as "No Stroke"
- Exploited class imbalance rather than learning minority patterns

This highlights a classic rare-event modeling issue:

> High accuracy does NOT imply good predictive performance.

---

## 📊 Statistical Validation

A one-sample z-test was performed to evaluate whether model accuracy exceeds random guessing (50%).

- **Z-statistic:** 64.61  
- **One-sided p-value:** < 1 × 10⁻³⁰⁸  

Result: The model performs significantly better than chance.

However, statistical significance does not imply clinical usefulness due to zero sensitivity.

---

## 🔍 Feature Importance (Top Predictors)

Top predictors identified:

1. Average Glucose Level
2. Age
3. BMI
4. Hypertension
5. Heart Disease

These align with established medical risk factors, confirming that the model captures meaningful structure in the majority class.

---

## 🧠 What This Project Demonstrates

This case study highlights:

- The danger of relying solely on accuracy
- The importance of sensitivity and recall in medical ML
- Challenges of imbalanced classification
- The need for cost-sensitive learning
- Why rare-event detection requires specialized methods

---

## 🚀 Future Improvements

To improve stroke detection:

- Apply SMOTE or oversampling
- Use class_weight adjustments
- Tune probability thresholds
- Optimize hyperparameters
- Evaluate ROC-AUC and PR curves
- Try XGBoost or cost-sensitive models

---
## 📁 Dataset Source

**Kaggle:**  
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

---

## 🏁 Conclusion

This project serves as a practical demonstration of **imbalanced classification challenges in healthcare analytics**.

While the Random Forest model achieved statistically significant accuracy, it failed to detect minority stroke cases, reinforcing the importance of:

- Proper metric selection  
- Rare-event modeling techniques  
- Interpreting ML outputs responsibly  

This case study emphasizes **methodological rigor over superficial performance metrics** and provides a foundation for future improvement in medical risk prediction systems.

---

⭐ If you found this case study insightful, feel free to star the repository.

