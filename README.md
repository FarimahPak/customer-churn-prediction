# Customer Churn Prediction

A machine learning project to predict customer churn using the IBM Telco Customer Churn dataset.

The project covers the end-to-end machine learning workflow, including data exploration, preprocessing, model training, evaluation, hyperparameter tuning, error analysis, probability calibration, and model interpretation.

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
   - Used feature scaling where appropriate.

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

The selected model is **Gradient Boosting Classifier** with a decision threshold of `0.4`.

### Validation Performance

| Metric | Score |
|---|---:|
| Precision | 0.637 |
| Recall | 0.662 |
| F1-score | 0.649 |
| ROC-AUC | 0.851 |
| PR-AUC | 0.671 |
| Brier Score | 0.1328 |

The model achieved an F1-score of approximately 0.649 on the validation set.

The decision threshold was selected based on validation performance, balancing precision and recall.

**Note:** These metrics represent validation performance, not an independent final test evaluation.

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
│   └── model_evaluation.md
│
├── models/
├── notebooks/
├── reports/
│   └── figures/
│
├── src/
│   └── churn_prediction/
│
├── tests/
│
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
- Jupyter Notebook
- Git and GitHub

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/FarimahPak/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the dataset in:

```text
data/raw/Telco-Customer-Churn.csv
```

5. Open the notebooks directory and run the notebooks in order.

## Limitations

- The analysis uses a single public dataset.
- Model performance may vary on other customer populations.
- SHAP interpretation was performed on a sample of the validation set.
- Model predictions indicate patterns in the available data and do not establish causal relationships.

## Future Improvements

- Perform a clean, independent final test evaluation.
- Improve recall for underrepresented customer groups.
- Explore additional feature engineering techniques.
- Experiment with other gradient boosting algorithms.
- Develop a reusable prediction pipeline.
- Add automated tests and model deployment capabilities.

## Author

**Farimah Pak**

Machine Learning & AI Enthusiast