# Disease Prediction Model Evaluation Report

## Executive Summary

This report summarizes the empirical performance of four Machine Learning models across three distinct medical datasets (Heart Disease, Diabetes, and Breast Cancer).

### Best Performing Models

- **Heart Disease**: `Random Forest` (Selected based on Composite F1 & ROC-AUC = 0.9163)
- **Diabetes**: `Random Forest` (Selected based on Composite F1 & ROC-AUC = 0.7282)
- **Breast Cancer**: `Logistic Regression` (Selected based on Composite F1 & ROC-AUC = 0.9866)

## Comprehensive Model Performance Comparison Table

| Dataset       | Model                        |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |
|:--------------|:-----------------------------|-----------:|------------:|---------:|-----------:|----------:|
| Heart Disease | Logistic Regression          |   0.868852 |    0.833333 | 0.892857 |   0.862069 |  0.963203 |
| Heart Disease | Support Vector Machine (SVM) |   0.852459 |    0.787879 | 0.928571 |   0.852459 |  0.962121 |
| Heart Disease | Random Forest                |   0.885246 |    0.83871  | 0.928571 |   0.881356 |  0.951299 |
| Heart Disease | XGBoost                      |   0.885246 |    0.818182 | 0.964286 |   0.885246 |  0.939394 |
| Diabetes      | Logistic Regression          |   0.701299 |    0.583333 | 0.518519 |   0.54902  |  0.809815 |
| Diabetes      | Support Vector Machine (SVM) |   0.707792 |    0.604651 | 0.481481 |   0.536082 |  0.812778 |
| Diabetes      | Random Forest                |   0.766234 |    0.695652 | 0.592593 |   0.64     |  0.816481 |
| Diabetes      | XGBoost                      |   0.74026  |    0.645833 | 0.574074 |   0.607843 |  0.825185 |
| Breast Cancer | Logistic Regression          |   0.982456 |    1        | 0.952381 |   0.97561  |  0.997685 |
| Breast Cancer | Support Vector Machine (SVM) |   0.973684 |    1        | 0.928571 |   0.962963 |  0.992725 |
| Breast Cancer | Random Forest                |   0.973684 |    1        | 0.928571 |   0.962963 |  0.994048 |
| Breast Cancer | XGBoost                      |   0.973684 |    1        | 0.928571 |   0.962963 |  0.992394 |

## Medical Domain Metric Interpretations

- **Recall (Sensitivity)**: Critical for disease screening. High recall minimizes **False Negatives** (failing to identify a patient with a condition).
- **Precision**: Minimizes **False Positives** (unnecessary anxiety and follow-up medical testing).
- **F1-Score & ROC-AUC**: Measures balanced diagnostic capability across varied risk thresholds.

## Disclaimer

> **Important Educational Disclaimer**: This evaluation is conducted for educational machine learning demonstration purposes only. Models must NOT be used for clinical diagnosis.
