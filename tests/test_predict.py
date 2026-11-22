
import numpy as np
import pandas as pd
import pytest

from churn_prediction.train import build_model
from churn_prediction import predict as predict_module


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


def test_model_pipeline_cleans_input_data():
    """Test that the model pipeline cleans raw customer data."""

    model = build_model()
    customers = make_customer_data()

    cleaner = model.named_steps["cleaner"]
    cleaned_data = cleaner.transform(customers)

    assert "customerID" not in cleaned_data.columns

    assert pd.api.types.is_numeric_dtype(
        cleaned_data["TotalCharges"]
    )

    assert cleaned_data["TotalCharges"].isna().sum() == 0

    assert cleaned_data["TotalCharges"].tolist() == [600.0, 0.0]

def test_full_pipeline_predicts_from_raw_customer_data():
    """Test predictions using raw customer data and the full pipeline."""

    from churn_prediction.train import train_model

    base_customers = make_customer_data()

    customers = pd.concat(
        [base_customers] * 4,
        ignore_index=True,
    )

    targets = pd.Series([0, 1] * 4)

    model = train_model(customers, targets)

    probabilities = model.predict_proba(customers)

    assert probabilities.shape == (8, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)

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

def test_predict_churn_validates_input_before_loading_model(monkeypatch):
    """Test that invalid input is rejected before loading the model."""

    def fail_if_model_loaded():
        pytest.fail("Model should not be loaded for invalid input.")

    monkeypatch.setattr(
        predict_module,
        "load_model",
        fail_if_model_loaded,
    )

    customers = make_customer_data()
    customers.loc[0, "Contract"] = "Invalid Contract"

    with pytest.raises(ValueError, match="Invalid values"):
        predict_module.predict_churn(customers)


def test_predict_churn_preserves_customer_data(monkeypatch):
    """Test that prediction results preserve customer information."""

    dummy_model = DummyModel()

    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda: dummy_model,
    )

    customers = make_customer_data()
    results = predict_module.predict_churn(customers)

    assert results["customerID"].tolist() == customers["customerID"].tolist()
    assert results["TotalCharges"].tolist() == [600.0, 0.0]
    assert "Churn_Probability" in results.columns
    assert "Churn_Prediction" in results.columns
    assert "Churn_Label" in results.columns