import joblib
import pandas as pd

from .config import MODEL_PATH, DECISION_THRESHOLD
from .preprocessing import validate_customer_data


def load_model():
    """Load the trained model from disk."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run the training module first."
        )

    return joblib.load(MODEL_PATH)


def predict_churn(customer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict churn probability and class for new customers."""

    customer_data = validate_customer_data(customer_data)
    model = load_model()

    probabilities = model.predict_proba(customer_data)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    results = customer_data.copy()
    results["Churn_Probability"] = probabilities
    results["Churn_Prediction"] = predictions
    results["Churn_Label"] = [
        "Likely to Churn" if prediction == 1 else "Likely to Stay"
        for prediction in predictions
    ]

    return results