# Logistic Regression Threshold Selection (Historical Experiment)

## Objective

Evaluate different classification thresholds using the validation set to understand the trade-off between Precision and Recall.

## Results

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.3 | 0.543 | 0.756 | 0.632 |
| 0.4 | 0.617 | 0.669 | 0.642 |
| 0.5 | 0.697 | 0.545 | 0.612 |
| 0.6 | 0.728 | 0.368 | 0.489 |
| 0.7 | 0.762 | 0.161 | 0.265 |

## Decision

A threshold of 0.4 was selected based on validation results.The test set was not used to select the threshold.

It achieved the highest F1-score among the evaluated thresholds while maintaining a reasonable balance between Precision and Recall.

## Considerations

- Lower thresholds improve Recall but increase False Positives.
- Higher thresholds improve Precision but increase False Negatives.
- The selected threshold is based on validation data, not test data.
- The existing test split was used for descriptive evaluation after threshold selection. Since it was consulted during earlier exploratory experiments, it is not considered an untouched independent test set.

## Selected Threshold

`0.4`
