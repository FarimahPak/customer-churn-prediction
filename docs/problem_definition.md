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

## 5. Modeling Approach

Logistic Regression was used as the initial baseline model to establish a reference performance.

Additional models, including Random Forest and Gradient Boosting, were evaluated during model development. Gradient Boosting was selected as the final model based on its validation performance.

## 6. Evaluation Metrics

The model will be evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Brier Score

Accuracy will also be reported, but it will not be the only evaluation metric.

## 7. Project Goal

The goal is to build a reproducible, well-documented, and tested machine learning pipeline that can predict customer churn and demonstrate practical machine learning engineering skills.

## 8. Project Limitations

The model's predictions are based on the available historical dataset and should not be interpreted as proof that a customer will churn.

The existing test split was consulted during earlier exploratory experiments. Therefore, its evaluation results should be considered descriptive rather than a fully independent estimate of generalization performance.

Further evaluation on an independent external dataset would provide stronger evidence of the model's ability to generalize to unseen customer data.