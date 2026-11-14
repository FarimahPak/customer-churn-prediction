# Problem Definition

## 1. Project Overview

This project aims to develop a machine learning model that predicts customer churn for a telecommunications company.

Customer churn occurs when a customer stops using a company's services. Predicting churn can help businesses identify customers who are more likely to leave and make better customer retention decisions.

## 2. Business Problem

Telecommunications companies may lose revenue when customers discontinue their services.

The business needs a way to identify customers who are at risk of churning so that appropriate retention strategies can be considered.

## 3. Machine Learning Problem

This project formulates customer churn prediction as a supervised binary classification problem.

Given historical customer information, the model will predict whether a customer belongs to one of two classes:

- `Yes`: The customer churned.
- `No`: The customer did not churn.

## 4. Target Variable

- **Target column:** `Churn`
- **Positive class:** `Yes`
- **Negative class:** `No`

## 5. Initial Model

Logistic Regression will be used as the initial baseline model.

More advanced models may be evaluated after establishing a reliable baseline.

## 6. Evaluation Metrics

The model will be evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Accuracy will also be reported, but it will not be the only evaluation metric.

## 7. Project Goal

The goal is to build a reproducible, well-documented, and tested machine learning pipeline that can predict customer churn and demonstrate practical machine learning engineering skills.

## 8. Project Limitations

The model's predictions will be based on the available historical dataset. Predictions should not be interpreted as proof that a customer will churn.