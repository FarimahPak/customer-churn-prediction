# Customer Churn Prediction

A machine learning project to predict customer churn using the IBM Telco Customer Churn dataset.

This project covers an end-to-end machine learning workflow, including exploratory data analysis, preprocessing, model development, evaluation, hyperparameter tuning, error analysis, probability calibration, and model interpretation.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help businesses better understand customer behavior and develop retention strategies.

The objective of this project is to build and evaluate machine learning models that predict whether a customer will leave the service.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information related to:

- Customer demographics
- Subscription services
- Account information
- Contract types
- Payment methods
- Customer tenure
- Churn status

The target variable is `Churn`:

- `Yes`: Customer churned
- `No`: Customer did not churn

## Project Workflow

1. **Exploratory Data Analysis**
   - Examined feature distributions and target imbalance.
   - Investigated relationships between customer attributes and churn.

2. **Data Preprocessing**
   - Handled missing and invalid values.
   - Converted numerical features to appropriate data types.
   - Removed irrelevant identifiers.
   - Applied One-Hot Encoding to categorical features.
   - Used numerical features without scaling in the final model pipeline.

3. **Model Development**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - PR-AUC
   - Brier Score

5. **Model Optimization**
   - Hyperparameter tuning using RandomizedSearchCV.
   - Decision threshold analysis.
   - Probability calibration experiments.

6. **Model Interpretation**
   - Built-in feature importance.
   - Permutation importance.
   - SHAP analysis.
   - Customer group analysis.
   - False positive and false negative investigation.

## Final Model

The selected model is a **Gradient Boosting Classifier** with a decision threshold of `0.4`.

### Model Configuration

| Parameter | Value |
|---|---:|
| n_estimators | 200 |
| learning_rate | 0.05 |
| max_depth | 2 |
| random_state | 42 |
| Decision threshold | 0.4 |

### Validation Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.8101 |
| Precision | 0.6367 |
| Recall | 0.6622 |
| F1-score | 0.6492 |
| ROC-AUC | 0.8510 |
| PR-AUC | 0.6707 |
| Brier Score | 0.1328 |

The decision threshold was selected based on validation F1-score, balancing precision and recall.

### Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.7857 |
| Precision | 0.5870 |
| Recall | 0.6497 |
| F1-score | 0.6168 |
| ROC-AUC | 0.8438 |
| PR-AUC | 0.6623 |
| Brier Score | 0.1361 |

Test confusion matrix:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 864 | 171 |
| Actual Yes | 131 | 243 |

The model correctly identified 243 churn customers and missed 131 actual churn cases.

**Evaluation note:** The test split was used in earlier exploratory model comparisons. Therefore, these results should not be considered a completely untouched, independent estimate of generalization performance. The decision threshold was selected using validation data and was not adjusted based on the latest test results.

## Model Interpretation

SHAP analysis was performed on 200 validation samples to better understand the model's predictions.

The most influential features included:

| Feature | Mean Absolute SHAP |
|---|---:|
| Contract: Month-to-month | 0.4935 |
| Tenure | 0.3781 |
| Internet Service: Fiber optic | 0.2715 |
| Contract: Two year | 0.1977 |
| Online Security: No | 0.1946 |
| Monthly Charges | 0.1783 |
| Tech Support: No | 0.1722 |

The results highlight the importance of contract type, customer tenure, and service-related features in the model's predictions.

SHAP values explain model behavior and should not be interpreted as causal relationships.

## Visualizations

### SHAP Feature Importance

![SHAP Feature Importance](reports/figures/shap_feature_importance.png)

### SHAP Summary Plot

![SHAP Summary Plot](reports/figures/shap_summary_plot.png)

### Confusion Matrix

![Confusion Matrix](reports/figures/confusion_matrix.png)

## Project Structure

```text
customer-churn-prediction/
│
├── configs/
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── decisions/
│   ├── data_dictionary.md
│   ├── model_evaluation.md
│   └── problem_definition.md
│
├── models/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_data_preprocessing.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   └── churn_prediction/
│       ├── data.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── evaluate.py
│       ├── final_evaluation.py
│       └── predict.py
│
├── tests/
├── .gitignore
├── PROJECT_GUIDE.md
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP
- Joblib
- Pytest
- pytest-cov
- Jupyter Notebook
- Git and GitHub

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/FarimahPak/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the IBM Telco Customer Churn CSV file in:

```text
data/raw/Telco-Customer-Churn.csv
```

### 5. Train the Model

```bash
python -m src.churn_prediction.train
```

This command trains the Gradient Boosting model and saves the trained model to:

```text
models/final_model.joblib
```

It also reports validation performance.

### 6. Run Final Test Evaluation

After training, run:

```bash
python -m src.churn_prediction.final_evaluation
```

This command loads the saved model and evaluates it on the recreated test split.

### 7. Run the Tests

```bash
python -m pytest -v
```

To run tests with coverage:

```bash
python -m pytest --cov=src/churn_prediction --cov-report=term-missing
```

The current test suite contains 24 tests and achieves approximately 98% code coverage.

## Limitations

- The analysis uses a single public dataset.
- Model performance may vary across other customer populations.
- The test split was consulted during earlier exploratory model comparisons.
- SHAP interpretation was performed on a sample of the validation set.
- Model predictions identify patterns in the available data and do not establish causal relationships.
- False positive and false negative predictions remain important limitations.

## Future Improvements

- Evaluate the model on a new, independent external dataset.
- Improve recall for underrepresented customer groups.
- Explore additional feature engineering techniques.
- Experiment with other gradient boosting algorithms.
- Develop a prediction API and deployment workflow.
- Add model monitoring and data drift detection.

## Author

**Farimah Pak**

Machine Learning & AI Enthusiast