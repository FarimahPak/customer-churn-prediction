# Data Dictionary

## Dataset Overview

The dataset contains 7,043 customer records and 21 original columns.

## Column Descriptions

| Column | Description | Data Type |
|---|---|---|
| customerID | Unique identifier assigned to each customer | Identifier |
| gender | Customer's gender | Categorical |
| SeniorCitizen | Indicates whether the customer is a senior citizen (0 or 1) | Binary categorical |
| Partner | Indicates whether the customer has a partner | Binary categorical |
| Dependents | Indicates whether the customer has dependents | Binary categorical |
| tenure | Number of months the customer has stayed with the company | Numerical |
| PhoneService | Indicates whether the customer has phone service | Binary categorical |
| MultipleLines | Indicates whether the customer has multiple phone lines | Categorical |
| InternetService | Type of internet service subscribed to | Categorical |
| OnlineSecurity | Indicates whether the customer has online security service | Categorical |
| OnlineBackup | Indicates whether the customer has online backup service | Categorical |
| DeviceProtection | Indicates whether the customer has device protection service | Categorical |
| TechSupport | Indicates whether the customer has technical support service | Categorical |
| StreamingTV | Indicates whether the customer has streaming TV service | Categorical |
| StreamingMovies | Indicates whether the customer has streaming movies service | Categorical |
| Contract | Customer's contract type | Categorical |
| PaperlessBilling | Indicates whether the customer uses paperless billing | Binary categorical |
| PaymentMethod | Customer's payment method | Categorical |
| MonthlyCharges | Monthly amount charged to the customer | Numerical |
| TotalCharges | Total amount charged to the customer | Numerical |
| Churn | Indicates whether the customer left the company (Yes/No) | Target |

## Numerical Feature Constraints

| Feature | Valid Range |
|---|---|
| tenure | 0–72 months |
| MonthlyCharges | 0–120 |
| TotalCharges | 0 or greater |

## Additional Notes

- `customerID` is an identifier and will be excluded from model features.
- `TotalCharges` contains 11 empty string values, all corresponding to records with `tenure = 0`. Empty values are converted to missing values during input validation and then filled with 0 during preprocessing.
- `tenure_group` is an additional feature created temporarily for EDA and is not part of the original dataset.
- `Churn` is the target variable for binary classification.
- The original dataset contains 20 input columns and one target column.
- After excluding `customerID`, 19 features are used as model inputs.