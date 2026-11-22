# Model Evaluation

## Logistic Regression Baseline (Historical Experiment)

### Experimental Setup

- Dataset: IBM Telco Customer Churn
- Training samples: 5,634
- Testing samples: 1,409
- Test size: 20%
- Random state: 42
- Stratification: Enabled
- Decision threshold: 0.5

### Evaluation Results

| Metric | Score |
|---|---:|
| Accuracy | 0.8055 |
| Precision (Churn) | 0.66 |
| Recall (Churn) | 0.56 |
| F1-score (Churn) | 0.60 |
| ROC-AUC | 0.8420 |
| PR-AUC | 0.6337 |

### Confusion Matrix

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 926 | 109 |
| Actual Yes | 165 | 209 |

### Initial Observations

- The baseline model achieves 80.55% accuracy.
- ROC-AUC of 0.8420 indicates good class discrimination.
- Recall for the positive class (Churn = Yes) is 0.56, meaning that approximately 44% of actual churn cases are missed.
- The model provides a reasonable starting point, but identifying more customers at risk of churn remains an improvement opportunity.

## Logistic Regression Threshold Analysis (Historical Experiment)

### Objective

Evaluate different classification thresholds using a validation set to understand the trade-off between Precision and Recall.

### Experimental Setup

- Validation samples: 1,127
- Thresholds evaluated: 0.3, 0.4, 0.5, 0.6, 0.7
- Evaluation metrics: Precision, Recall, F1-score
- Test set was not used for threshold selection.

### Results

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.3 | 0.543 | 0.756 | 0.632 |
| 0.4 | 0.617 | 0.669 | 0.642 |
| 0.5 | 0.697 | 0.545 | 0.612 |
| 0.6 | 0.728 | 0.368 | 0.489 |
| 0.7 | 0.762 | 0.161 | 0.265 |

### Selected Threshold

The decision threshold was selected based on validation F1-score. The test set was not used for threshold selection.

A threshold of 0.4 achieved the highest F1-score among the tested thresholds and was selected for the subsequent Logistic Regression evaluation.

## Logistic Regression Evaluation at Threshold 0.4 (Historical Experiment)

### Experimental Setup

- Model: Logistic Regression
- Training samples: 5,634
- Testing samples: 1,409
- Decision threshold: 0.4
- Training strategy: Retrained using the complete training set after threshold selection.

### Evaluation Results

| Metric | Score |
|---|---:|
| Accuracy | 0.777 |
| Precision (Churn) | 0.568 |
| Recall (Churn) | 0.668 |
| F1-score (Churn) | 0.614 |
| ROC-AUC | 0.842 |
| PR-AUC | 0.634 |

### Confusion Matrix

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 845 | 190 |
| Actual Yes | 124 | 250 |

### Observations

- Recall increased from 0.56 at the baseline threshold of 0.5 to 0.67 at the selected threshold of 0.4.
- The model identified 41 additional churn customers compared with the baseline threshold.
- False positives increased from 109 to 190, while Accuracy decreased.
- ROC-AUC and PR-AUC remained unchanged because threshold adjustment does not affect the predicted probabilities.
- The threshold of 0.4 favors identifying more potential churn customers at the cost of lower Precision.

## Logistic Regression Experiment Conclusion

The Logistic Regression model provided a useful baseline for customer churn prediction. Applying the threshold of 0.4, selected using validation data, increased Recall on the existing test split compared with the baseline threshold of 0.5, while Precision and Accuracy decreased.

Because the test split was consulted during earlier exploratory experiments, these results should be treated as descriptive comparisons rather than an independent estimate of generalization performance.

The Logistic Regression model was later compared with other candidates, and Gradient Boosting achieved the highest validation F1-score among the tested models.

## Planned Follow-up Experiments (Historical)

The following experiments were identified as potential next steps during the initial Logistic Regression analysis. They were subsequently explored in the experiments documented below.

- Perform error analysis on False Positives and False Negatives.
- Evaluate Logistic Regression with class weighting.
- Compare additional classification algorithms.
- Investigate feature importance and model interpretability.

## Logistic Regression Error Analysis (Historical Experiment)

### False Negative Analysis by Contract Type

The Logistic Regression model's False Negatives were analyzed to identify patterns among customers who actually churned but were predicted as non-churning.

| Contract Type | Actual Churners | False Negatives | Detected Churners | Recall |
|---|---:|---:|---:|---:|
| Month-to-month | 329 | 83 | 246 | 0.748 |
| One year | 36 | 32 | 4 | 0.111 |
| Two year | 9 | 9 | 0 | 0.000 |

### Observations

- The model performs substantially better on month-to-month customers.
- Recall is considerably lower for customers with one-year and two-year contracts.
- Long-term contract customers are overrepresented among False Negatives relative to their share of actual churners.
- The small number of long-term contract churn cases limits the reliability of these subgroup estimates.

### Initial Interpretation

The model may be relying heavily on patterns associated with month-to-month contracts when identifying potential churn customers.

Although long-term contracts are generally associated with lower churn rates, some customers in these groups still churn.

This finding suggests that subgroup performance should be considered alongside overall evaluation metrics.

Further investigation is needed before drawing definitive conclusions or changing the model.

### False Positive Analysis

False Positives were analyzed to identify patterns among customers predicted to churn who actually remained with the company.

#### Numerical Features

| Feature | Mean |
|---|---:|
| Tenure | 16.88 |
| MonthlyCharges | 78.78 |
| TotalCharges | 1489.09 |

#### Categorical Features

**Contract Distribution**

| Contract Type | False Positives |
|---|---:|
| Month-to-month | 97.37% |
| One year | 2.63% |
| Two year | 0.00% |

**Internet Service Distribution**

| Internet Service | False Positives |
|---|---:|
| Fiber optic | 79.47% |
| DSL | 20.53% |
| No internet service | 0.00% |

#### Comparison with All Actual Non-Churners

| Feature | Category | All Non-Churners | False Positives |
|---|---|---:|---:|
| Contract | Month-to-month | 42.90% | 97.37% |
| Contract | One year | 25.51% | 2.63% |
| Contract | Two year | 31.59% | 0.00% |
| Internet Service | Fiber optic | 34.88% | 79.47% |
| Internet Service | DSL | 37.39% | 20.53% |
| Internet Service | No internet | 27.73% | 0.00% |

#### Observations

- False Positives are heavily concentrated among month-to-month customers.
- Fiber optic customers are also overrepresented among False Positives.
- The model appears to have difficulty distinguishing loyal customers from churned customers when they share characteristics associated with higher churn rates.

#### Initial Interpretation

The model may be relying on general churn patterns that do not apply to every individual customer.

These findings suggest that overall performance metrics alone do not fully describe model behavior across different customer groups.

Further investigation is needed before deciding whether feature engineering, class weighting, or a different classification algorithm could address these errors.

### False Negative Analysis by Internet Service

The model's Recall was also evaluated across different Internet Service categories.

| Internet Service | Actual Churners | False Negatives | Detected Churners | Recall |
|---|---:|---:|---:|---:|
| DSL | 97 | 53 | 44 | 0.454 |
| Fiber optic | 252 | 46 | 206 | 0.817 |
| No internet service | 25 | 25 | 0 | 0.000 |

#### Observations

- The model achieves its highest Recall among Fiber optic customers (0.817).
- Recall is considerably lower among DSL customers (0.454).
- None of the 25 actual churners without Internet Service were detected.
- The small sample size of the No internet service group limits the reliability of its Recall estimate.

#### Interpretation

The model's ability to identify churn customers varies considerably across Internet Service categories.

Although Fiber optic customers are associated with higher churn rates in the dataset, the model performs less effectively for DSL and customers without Internet Service.

This suggests that subgroup-level performance should be considered when evaluating the model, rather than relying exclusively on overall metrics.

## Model Interpretability

### Logistic Regression Coefficients

The model coefficients were examined to understand how individual features influence churn predictions.

Key positive coefficients:
- InternetService_Fiber optic: 0.648
- Contract_Month-to-month: 0.586
- TotalCharges: 0.533
- StreamingMovies_Yes: 0.212
- StreamingTV_Yes: 0.212

Key negative coefficients:
- tenure: -1.255
- Contract_Two year: -0.759
- InternetService_DSL: -0.644
- MonthlyCharges: -0.601

Positive coefficients increase the model's predicted log-odds of churn, while negative coefficients decrease them, holding other features constant.

### Permutation Feature Importance

Permutation Importance was calculated on the validation set using ROC-AUC as the scoring metric, with 10 repetitions.

| Feature | Importance Mean | Importance Std |
|---|---:|---:|
| tenure | 0.152548 | 0.014467 |
| InternetService | 0.042831 | 0.006053 |
| Contract | 0.038143 | 0.007900 |
| TotalCharges | 0.021076 | 0.005371 |
| MonthlyCharges | 0.015736 | 0.003455 |
| StreamingTV | 0.006546 | 0.001866 |
| StreamingMovies | 0.006067 | 0.001556 |
| PaperlessBilling | 0.005997 | 0.001798 |
| PaymentMethod | 0.005466 | 0.001222 |
| OnlineSecurity | 0.005287 | 0.001434 |

### Key Findings

- Tenure is the most influential feature according to Permutation Importance.
- InternetService and Contract are the second and third most important features.
- The results are broadly consistent with the Logistic Regression coefficient analysis.
- Partner, gender, and PhoneService have almost no measured importance in this experiment.

Permutation Importance measures the impact of disrupting a feature on model performance. It does not establish causality, and correlated features may share predictive information.

These findings describe the current Logistic Regression model and should not be interpreted as universal measures of customer churn behavior.

## Logistic Regression Class Weighting Experiment (Historical)

### Objective

Evaluate whether assigning higher weights to the minority class improves churn detection.

### Experimental Setup

- Model: Logistic Regression
- Class weighting: balanced
- Validation and test sets: Same splits as previous experiments
- Thresholds evaluated: 0.3, 0.4, 0.5, 0.6, 0.7
- Threshold selection: Based on validation F1-score

### Validation Threshold Results

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.3 | 0.421 | 0.916 | 0.577 |
| 0.4 | 0.466 | 0.873 | 0.608 |
| 0.5 | 0.524 | 0.799 | 0.633 |
| 0.6 | 0.573 | 0.709 | 0.634 |
| 0.7 | 0.657 | 0.595 | 0.625 |

A threshold of 0.6 was selected because it achieved the highest validation F1-score among the evaluated thresholds.

### Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.762 |
| Precision | 0.540 |
| Recall | 0.706 |
| F1-score | 0.612 |
| ROC-AUC | 0.842 |
| PR-AUC | 0.633 |

### Confusion Matrix

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 810 | 225 |
| Actual Yes | 110 | 264 |

### Comparison with the Standard Logistic Regression

- Class weighting increased Recall from 0.668 to 0.706 on the test set.
- False Negatives decreased from 124 to 110.
- False Positives increased from 190 to 225.
- F1-score remained nearly unchanged (0.614 vs. 0.612).

### Conclusion

Class weighting improved the model's ability to identify churn customers, but this came at the cost of additional False Positive predictions.

The standard Logistic Regression with a threshold of 0.4 remains the preferred candidate based on validation F1-score and the observed test results.

The balanced model remains a useful alternative when identifying more potential churn customers is prioritized over minimizing unnecessary retention interventions.

Note: Test results are exploratory comparisons because the test set has been consulted during multiple model experiments. They should not be treated as an entirely untouched final evaluation.

## 7. Gradient Boosting Evaluation

### Initial Model

A Gradient Boosting Classifier was trained using the existing preprocessing pipeline.

Validation performance at threshold 0.5:

| Metric | Score |
|---|---:|
| Accuracy | 0.8101 |
| Precision | 0.7014 |
| Recall | 0.4950 |
| F1-score | 0.5804 |
| ROC-AUC | 0.8510 |
| PR-AUC | 0.6707 |
| Brier Score | 0.1328 |

### Threshold Selection

Threshold tuning was performed on the validation set.

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.3 | 0.553 | 0.753 | 0.637 |
| 0.4 | 0.637 | 0.662 | **0.649** |
| 0.5 | 0.701 | 0.495 | 0.580 |
| 0.6 | 0.714 | 0.351 | 0.471 |
| 0.7 | 0.753 | 0.214 | 0.333 |

Threshold 0.4 achieved the highest F1-score among the tested thresholds.

Confusion matrix at threshold 0.4:

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 715 | 113 |
| Actual Churn | 101 | 198 |

### Hyperparameter Tuning

RandomizedSearchCV was used with:
- 20 parameter combinations
- 3-fold cross-validation
- Average Precision (PR-AUC) as the optimization metric

Best parameters:

- `n_estimators`: 100
- `learning_rate`: 0.1
- `max_depth`: 1
- `min_samples_leaf`: 6

Best cross-validation PR-AUC: 0.6632.

The tuned model did not improve validation performance compared with the initial Gradient Boosting model. Its best tested F1-score was 0.644 at threshold 0.4, compared with 0.649 for the initial model.

The initial Gradient Boosting model was therefore retained as the stronger candidate.

## 8. Feature Importance Analysis

Two feature importance methods were examined.

### Built-in Feature Importance

The most important features included:
- Contract (Month-to-month)
- Tenure
- Internet Service (Fiber optic)
- Online Security
- Tech Support

The Month-to-month contract category had the highest individual importance.

### Permutation Importance

Permutation importance was calculated on the validation set using Average Precision as the scoring metric, with 10 repetitions.

Top features:

| Feature | Mean Importance | Std |
|---|---:|---:|
| tenure | 0.1001 | 0.0161 |
| Contract | 0.0893 | 0.0132 |
| InternetService | 0.0486 | 0.0076 |
| TotalCharges | 0.0413 | 0.0082 |
| MonthlyCharges | 0.0279 | 0.0125 |
| TechSupport | 0.0177 | 0.0079 |
| OnlineSecurity | 0.0162 | 0.0069 |

Both methods highlighted tenure, contract type, and internet service as important predictors.

Feature importance indicates predictive relevance, not causation or the direction of a feature's effect.

## 9. Churn Rate Analysis by Customer Groups

Churn rates were calculated using the training data.

| Feature | Category | Customers | Churn Rate |
|---|---|---:|---:|
| Contract | Month-to-month | 3102 | 42.75% |
| Contract | One year | 1173 | 11.08% |
| Contract | Two year | 1359 | 2.87% |
| InternetService | DSL | 1937 | 18.69% |
| InternetService | Fiber optic | 2483 | 42.09% |
| InternetService | No | 1214 | 7.25% |
| OnlineSecurity | No | 2797 | 41.94% |
| OnlineSecurity | Yes | 1623 | 14.42% |
| TechSupport | No | 2771 | 41.75% |
| TechSupport | Yes | 1649 | 15.16% |

### Key Observations

- Month-to-month customers have substantially higher churn rates than customers with longer contracts.
- Fiber optic customers show a higher churn rate than DSL customers.
- Customers without Online Security or Tech Support have higher observed churn rates than customers with these services.

These are descriptive associations and should not be interpreted as causal relationships.

## 10. Model Comparison

The best tested threshold for each model was selected based on validation F1-score.

| Model | Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.4 | 0.617 | 0.669 | 0.642 |
| Tuned Random Forest | 0.6 | 0.610 | 0.686 | 0.646 |
| Gradient Boosting | 0.4 | 0.637 | 0.662 | **0.649** |

Gradient Boosting achieved the highest validation F1-score among the tested candidates, while Random Forest achieved the highest recall.

The differences are small. The final model selection should also consider interpretability, probability quality, and the business cost of false positives and false negatives.

**Evaluation note:** Model selection and threshold comparisons are based on validation data. The test split was consulted during earlier exploratory experiments and should therefore be interpreted as a descriptive evaluation rather than an independent estimate of generalization performance.

## 11. Gradient Boosting Error Analysis

### 11.1 Overall Error Distribution

The initial Gradient Boosting model was evaluated on the validation set using a classification threshold of 0.40.

| Error Type | Count |
|---|---:|
| Correct Predictions | 913 |
| False Positives | 113 |
| False Negatives | 101 |
| Total | 1,127 |

The model missed 101 out of 299 actual churners, resulting in a recall of approximately 66.2%.

### 11.2 Error Analysis by Customer Segment

**Contract Type**

| Contract | Actual Churners | False Negatives | Recall |
|---|---:|---:|---:|
| Month-to-month | 259 | 61 | 76.4% |
| One year | 29 | 29 | 0.0% |
| Two year | 11 | 11 | 0.0% |

The model failed to identify any churners in the one-year and two-year contract groups in the validation set.

**Internet Service**

| Internet Service | Actual Churners | False Negatives | Recall |
|---|---:|---:|---:|
| DSL | 69 | 46 | 33.3% |
| Fiber optic | 215 | 42 | 80.5% |
| No internet service | 15 | 13 | 13.3% |

The model performed considerably better for Fiber optic customers than for DSL customers and customers without internet service.

### 11.3 Predicted Churn Probability Analysis

Mean predicted churn probabilities among actual churners:

| Segment | Group | Mean Probability |
|---|---|---:|
| Contract | Month-to-month | 0.537 |
| Contract | One year | 0.152 |
| Contract | Two year | 0.077 |
| Internet Service | DSL | 0.337 |
| Internet Service | Fiber optic | 0.550 |
| Internet Service | No internet service | 0.191 |

The model assigned relatively low churn probabilities to actual churners with long-term contracts. Even the maximum predicted probability was below the selected 0.40 threshold for both long-term contract groups.

This suggests that the missed predictions were not simply borderline cases near the decision threshold.

### 11.4 False Negative Characteristics

A comparison between correctly detected churners and false negatives revealed the following patterns:

| Feature | Correctly Detected Churners | False Negatives |
|---|---:|---:|
| Mean Tenure | 13.21 | 32.18 |
| Mean Monthly Charges | 80.94 | 68.35 |
| Mean Total Charges | 1,212.19 | 2,671.85 |
| Mean Predicted Churn Probability | 0.62 | 0.21 |

False negatives generally had longer customer tenure, lower monthly charges, and higher total charges than correctly detected churners.

Within the month-to-month group, missed churners also had longer average tenure and lower monthly charges:

| Prediction Status | Mean Tenure | Mean Monthly Charges |
|---|---:|---:|
| Correct | 13.21 | 80.94 |
| False Negative | 19.08 | 57.63 |

All actual churners with one-year and two-year contracts were missed. Their mean tenure was 49.24 and 59.82 months, respectively.

### 11.5 Key Findings and Limitations

The error analysis highlights several limitations:

- The model has difficulty identifying churn among customers with long-term contracts.
- Recall varies substantially across internet service categories.
- Customers with longer tenure are more frequently missed.
- The model tends to underestimate churn probabilities for some customer segments.

These findings are descriptive and do not establish causal relationships. Some segments, particularly two-year contract customers and customers without internet service, contain relatively few validation examples. Their recall estimates should therefore be interpreted cautiously.

Further investigation could explore segment-specific performance, probability calibration, class weighting, and alternative modeling approaches.

## 12. Probability Calibration

### 12.1 Calibration Evaluation

Probability calibration was evaluated to determine whether predicted churn probabilities correspond to observed churn frequencies.

The initial Gradient Boosting model achieved a Brier Score of 0.1328 on the validation set.

A quantile-based calibration analysis showed that the model was reasonably calibrated across several probability ranges, although some deviations were observed.

The largest underestimation occurred in the probability bin with a mean predicted probability of 0.566 and an observed churn rate of 0.690.

### 12.2 Calibration Methods

Two calibration methods were evaluated using `CalibratedClassifierCV` with five-fold cross-validation:

- Sigmoid (Platt Scaling)
- Isotonic Regression

Both methods were fitted using the training subset and evaluated on the validation set.

| Model | Brier Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Original Gradient Boosting | 0.1328 | 0.8510 | 0.6707 |
| Sigmoid Calibration | 0.1327 | 0.8514 | 0.6688 |
| Isotonic Calibration | 0.1332 | 0.8500 | 0.6676 |

### 12.3 Calibration Gap Comparison

Mean Absolute Calibration Gap was also calculated using quantile-based bins:

| Model | Mean Absolute Calibration Gap |
|---|---:|
| Original | 0.0340 |
| Sigmoid | 0.0391 |
| Isotonic | 0.0314 |

The Isotonic model achieved the lowest mean absolute calibration gap, while the Sigmoid model achieved the lowest Brier Score.

However, the calibration gap comparison should be interpreted cautiously because quantile bins were independently generated for each model.

### 12.4 Final Calibration Decision

Neither calibration method demonstrated a consistent improvement across all evaluation metrics.

The original Gradient Boosting model was retained as the selected model because:

- It achieved the highest PR-AUC.
- Its Brier Score was nearly identical to the Sigmoid-calibrated model.
- Calibration improvements from the alternative methods were marginal or inconsistent.

The calibration experiments were retained as comparative experiments rather than replacing the original model.

## 13. Model Interpretation with SHAP

### 13.1 Objective

SHAP (SHapley Additive exPlanations) was used to interpret the final Gradient Boosting model and understand how individual features contribute to its predictions.

The analysis focused on:
- Identifying the most influential features.
- Understanding the direction and magnitude of feature contributions.
- Exploring how feature contributions vary across customer groups.

### 13.2 Methodology

The final Gradient Boosting classifier was selected for SHAP analysis.

Since the model was implemented as a Scikit-learn Pipeline, the validation data was first transformed using the fitted preprocessing pipeline. SHAP values were then calculated using `shap.TreeExplainer` on the fitted Gradient Boosting classifier.

Analysis configuration:

| Setting | Value |
|---|---|
| Explainer | TreeExplainer |
| SHAP version | 0.54.1 |
| Dataset | Validation set |
| Samples analyzed | 200 |
| Transformed features | 46 |
| Model | Gradient Boosting |

The first 200 validation samples were used for this analysis.

### 13.3 Global Feature Importance

Mean absolute SHAP values were used to measure the overall influence of each feature on model predictions.

Top 15 features:

| Feature | Mean Absolute SHAP |
|---|---:|
| Contract: Month-to-month | 0.4935 |
| Tenure | 0.3781 |
| Internet Service: Fiber optic | 0.2715 |
| Contract: Two year | 0.1977 |
| Online Security: No | 0.1946 |
| Monthly Charges | 0.1783 |
| Tech Support: No | 0.1722 |
| Total Charges | 0.1452 |
| Multiple Lines: No | 0.1335 |
| Payment Method: Electronic check | 0.1234 |
| Online Backup: No | 0.0782 |
| Streaming TV: Yes | 0.0682 |
| Paperless Billing: No | 0.0631 |
| Streaming Movies: Yes | 0.0538 |
| Paperless Billing: Yes | 0.0328 |

The results indicate that contract type and customer tenure are among the most influential features in the model.

Internet service type, online security, technical support, and billing-related features also contribute to the predictions.

### 13.4 SHAP Direction and Distribution

Mean SHAP values were also examined to understand the average direction of feature contributions.

Some features showed mean values close to zero despite having relatively high mean absolute SHAP values.

For example:

- Fiber optic Internet Service: Mean SHAP = -0.0027
- Mean Absolute SHAP = 0.2715

This indicates that the feature can contribute positively or negatively across different customers, causing its average signed contribution to approach zero.

Therefore, mean absolute SHAP values are more appropriate for measuring global importance, while signed SHAP values provide additional information about contribution direction.

### 13.5 Tenure Analysis

Tenure was further analyzed by grouping customers into four membership-duration ranges.

| Tenure Group | Customer Count | Mean SHAP | Mean Absolute SHAP |
|---|---:|---:|---:|
| 0–12 months | 64 | 0.4316 | 0.4552 |
| 13–24 months | 27 | -0.0528 | 0.1186 |
| 25–48 months | 54 | -0.2838 | 0.2838 |
| 49–72 months | 55 | -0.5084 | 0.5084 |

The results show a clear pattern:

- Customers with 0–12 months of tenure had a positive average SHAP contribution.
- The average contribution became negative for customers with more than 12 months of tenure.
- Customers with 49–72 months of tenure had the strongest negative average contribution.

This suggests that the model generally associates longer customer tenure with lower churn predictions.

### 13.6 Tenure and Contract Analysis

The contribution of Tenure was also examined across contract types.

| Contract Type | Customer Count | Mean Tenure SHAP | Mean Absolute Tenure SHAP |
|---|---:|---:|---:|
| Month-to-month | 115 | 0.1196 | 0.3787 |
| One year | 40 | -0.2824 | 0.3132 |
| Two year | 45 | -0.4342 | 0.4342 |

Tenure contributions differed across contract groups.

The average Tenure SHAP contribution was positive for month-to-month customers and negative for customers with one-year and two-year contracts.

This is a descriptive comparison and does not establish a causal relationship or an independent interaction effect.

### 13.7 Contract Feature Contributions

The SHAP contributions of individual one-hot encoded contract features were also examined.

| Feature | Mean SHAP | Mean Absolute SHAP |
|---|---:|---:|
| Month-to-month | 0.0479 | 0.4935 |
| One year | -0.0061 | 0.0237 |
| Two year | 0.0196 | 0.1977 |

Month-to-month had the highest overall contribution magnitude among the contract categories.

Signed mean values should be interpreted cautiously because contributions can vary across individual customers and correlated features.

### 13.8 Key Findings

The SHAP analysis highlighted the following findings:

1. Contract type, particularly month-to-month contracts, is highly influential in the model's predictions.
2. Tenure is another major contributor, with substantially different contribution patterns across tenure groups.
3. Fiber optic Internet Service, lack of Online Security, and lack of Tech Support also have notable contributions.
4. Feature contributions vary across customers, meaning that global importance alone cannot fully explain individual predictions.
5. The findings are broadly consistent with the earlier feature-importance and error-analysis results.

### 13.9 Limitations

- SHAP analysis was performed on the first 200 validation samples rather than the entire validation set.
- The results may not fully represent the distribution of all validation customers.
- Mean absolute SHAP values measure contribution magnitude, not whether a feature increases or decreases churn predictions.
- SHAP values explain model behavior and do not establish causal relationships.
- The TreeExplainer output represents the model's raw output scale in this analysis, not directly calibrated churn probabilities.

Overall, SHAP provided additional insight into the behavior of the final Gradient Boosting model and complemented the previous feature-importance and error-analysis methods.


## 14. Final Gradient Boosting Test Evaluation

### 14.1 Evaluation Setup

The selected Gradient Boosting model was evaluated on the test split using the previously selected decision threshold.

| Setting | Value |
|---|---|
| Model | Gradient Boosting Classifier |
| Training samples | 4,507 |
| Validation samples | 1,127 |
| Test samples | 1,409 |
| Decision threshold | 0.4 |
| Number of estimators | 200 |
| Learning rate | 0.05 |
| Maximum depth | 2 |
| Random state | 42 |

The model was loaded from the saved model artifact and was not retrained during final evaluation.

The same deterministic data split was recreated using the existing splitting procedure.

### 14.2 Test Evaluation Results

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.8101 | 0.7857 |
| Precision | 0.6367 | 0.5870 |
| Recall | 0.6622 | 0.6497 |
| F1-score | 0.6492 | 0.6168 |
| ROC-AUC | 0.8510 | 0.8438 |
| PR-AUC | 0.6707 | 0.6623 |
| Brier Score | 0.1328 | 0.1361 |

### 14.3 Test Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 864 | 171 |
| Actual Churn | 131 | 243 |

The model correctly identified 243 churn customers and missed 131 actual churn cases.

It also generated 171 false positive predictions.

### 14.4 Observations

- Test ROC-AUC (0.8438) remained close to validation ROC-AUC (0.8510).
- Recall decreased slightly from 0.6622 to 0.6497.
- Precision decreased from 0.6367 to 0.5870.
- F1-score decreased from 0.6492 to 0.6168.
- The Brier Score increased slightly from 0.1328 to 0.1361.

Overall, the model demonstrated reasonably consistent ranking performance between validation and test data, although classification performance was lower on the test split.

### 14.5 Evaluation Limitations

The test split was used in earlier exploratory model comparisons and experiments documented in this report.

Therefore, this evaluation should not be interpreted as a completely untouched, independent estimate of generalization performance.

The reported results describe the performance of the selected Gradient Boosting model on the existing test split. Further evaluation on a new external dataset would provide stronger evidence of generalization.

The decision threshold of 0.4 was selected using validation data and was not adjusted based on the latest test results.

### 14.6 Final Conclusion

The selected Gradient Boosting model was retained because it achieved the highest validation F1-score among the compared candidates.

At the selected threshold of 0.4, the model achieved a test F1-score of 0.6168 and a ROC-AUC of 0.8438.

The results indicate that the model can identify a meaningful proportion of churn customers, but false positive and false negative predictions remain important limitations.

Future improvements could focus on reducing missed churn cases, investigating segment-level performance, and evaluating the model on an independent external dataset.
