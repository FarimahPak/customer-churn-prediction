
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

