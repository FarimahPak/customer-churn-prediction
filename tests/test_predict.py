
import numpy as np
import pandas as pd
import pytest

from src.churn_prediction import predict as predict_module


class DummyModel:
    """A lightweight model used to test prediction behavior."""

    def __init__(self):
        self.received_data = None

    def predict_proba(self, X):
        self.received_data = X.copy()

        return np.array([
            [0.30, 0.70],
            [0.75, 0.25],
        ])


def make_customer_data():
    """Create synthetic customer data for prediction tests."""

    return pd.DataFrame({
        "customerID": ["C001", "C002"],
        "tenure": [12, 24],
        "MonthlyCharges": [50.0, 80.0],
        "TotalCharges": ["600.0", " "],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["Yes", "No"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
        ],
    })


def test_predict_churn_returns_expected_results(monkeypatch):
    """Test churn probabilities, predictions, and labels."""

    model = DummyModel()

    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda: model,
    )

    customers = make_customer_data()

    results = predict_module.predict_churn(customers)

    assert results["Churn_Prediction"].tolist() == [1, 0]

    assert results["Churn_Label"].tolist() == [
        "Likely to Churn",
        "Likely to Stay",
    ]

    assert results["Churn_Probability"].tolist() == [0.70, 0.25]


def test_predict_churn_cleans_input_data(monkeypatch):
    """Test that input data is cleaned before prediction."""

    model = DummyModel()

    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda: model,
    )

    customers = make_customer_data()

    predict_module.predict_churn(customers)

    assert "customerID" not in model.received_data.columns
    assert pd.api.types.is_numeric_dtype(
        model.received_data["TotalCharges"]
    )
    assert model.received_data["TotalCharges"].isna().sum() == 0
    assert model.received_data["TotalCharges"].tolist() == [600.0, 0.0]


def test_load_model_file_not_found(monkeypatch, tmp_path):
    """Test error when the saved model does not exist."""

    missing_model_path = tmp_path / "missing_model.joblib"

    monkeypatch.setattr(
        predict_module,
        "MODEL_PATH",
        missing_model_path,
    )

    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        predict_module.load_model()
