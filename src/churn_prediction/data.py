
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

TARGET_COLUMN = "Churn"

REQUIRED_COLUMNS = {
    "customerID",
    "Churn",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
}


def load_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features from the churn target."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Target column contains missing or unexpected values.")

    return X, y.astype(int)
