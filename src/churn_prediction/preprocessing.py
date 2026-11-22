
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERICAL_RANGES = {
    "tenure": (0, 72),
    "MonthlyCharges": (0, 120),
    "TotalCharges": (0, float("inf")),
}

CATEGORICAL_VALUES = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    "Partner": {"No", "Yes"},
    "Dependents": {"No", "Yes"},
    "PhoneService": {"No", "Yes"},
    "MultipleLines": {"No", "No phone service", "Yes"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"No", "No internet service", "Yes"},
    "OnlineBackup": {"No", "No internet service", "Yes"},
    "DeviceProtection": {"No", "No internet service", "Yes"},
    "TechSupport": {"No", "No internet service", "Yes"},
    "StreamingTV": {"No", "No internet service", "Yes"},
    "StreamingMovies": {"No", "No internet service", "Yes"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"No", "Yes"},
    "PaymentMethod": {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    },
}

def validate_customer_data(X: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw customer input."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Customer data must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("Customer data must contain at least one row.")

    if X.columns.duplicated().any():
        duplicate_columns = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns found: {duplicate_columns}")

    required_columns = set(NUMERICAL_RANGES) | set(CATEGORICAL_VALUES)
    optional_columns = {"customerID"}

    missing_columns = required_columns - set(X.columns)
    extra_columns = set(X.columns) - required_columns - optional_columns

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if extra_columns:
        raise ValueError(
            f"Unexpected columns: {sorted(extra_columns)}"
        )

    X = X.copy()

    if "customerID" in X.columns:
        customer_ids = X["customerID"]

        if customer_ids.isna().any():
            raise ValueError(
                "Column 'customerID' cannot contain missing values."
            )

        if not customer_ids.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(
                "Column 'customerID' must contain non-empty strings."
            )

    for column, allowed_values in CATEGORICAL_VALUES.items():
        invalid_values = set(X[column].dropna().unique()) - allowed_values

        if invalid_values:
            raise ValueError(
                f"Invalid values in '{column}': "
                f"{sorted(invalid_values, key=str)}"
            )

    for column, (minimum, maximum) in NUMERICAL_RANGES.items():
        original_values = X[column]

        if column == "TotalCharges":
            normalized_values = original_values.astype("string").str.strip()
            values = pd.to_numeric(normalized_values, errors="coerce")

            invalid_conversion = (
                original_values.notna()
                & normalized_values.ne("").fillna(False)
                & values.isna()
            )

            if invalid_conversion.any():
                raise ValueError(
                    f"Column '{column}' contains non-numeric values."
                )

        else:
            values = pd.to_numeric(original_values, errors="coerce")

            if values.isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing or non-numeric values."
                )

        if not values.dropna().between(minimum, maximum).all():
            raise ValueError(
                f"Column '{column}' must contain values between "
                f"{minimum} and {maximum}."
            )

        if column == "tenure":
            if not values.dropna().mod(1).eq(0).all():
                raise ValueError(
                    "Column 'tenure' must contain integer values."
                )

        if column == "TotalCharges":
            values = values.fillna(0)

        X[column] = values

    return X

def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customer features before preprocessing."""

    X = X.copy()

    X = X.drop(columns=["customerID"], errors="ignore")

    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"].astype(str).str.strip(),
        errors="coerce",
    )

    X["TotalCharges"] = X["TotalCharges"].fillna(0)

    return X



def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for the final Gradient Boosting model."""

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor