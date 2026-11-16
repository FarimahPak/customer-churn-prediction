
import pandas as pd

from src.churn_prediction.data import load_data, split_features_target
from src.churn_prediction.preprocessing import (
    clean_features,
    build_preprocessor,
)


def test_clean_features_removes_customer_id():
    """Test that customerID is removed."""

    X = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "TotalCharges": ["100.5", "200.0"],
    })

    cleaned = clean_features(X)

    assert "customerID" not in cleaned.columns


def test_clean_features_converts_and_fills_total_charges():
    """Test numeric conversion and missing value handling."""

    X = pd.DataFrame({
        "customerID": ["C001", "C002", "C003"],
        "TotalCharges": ["100.5", " ", "300.0"],
    })

    cleaned = clean_features(X)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].tolist() == [100.5, 0.0, 300.0]
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_clean_features_does_not_modify_original_data():
    """Test that cleaning does not change the original DataFrame."""

    X = pd.DataFrame({
        "customerID": ["C001"],
        "TotalCharges": ["100.5"],
    })

    original = X.copy()

    clean_features(X)

    pd.testing.assert_frame_equal(X, original)


def test_preprocessor_output_shape():
    """Test the transformed feature shape on the full dataset."""

    df = load_data()
    X, _ = split_features_target(df)
    X = clean_features(X)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape == (7043, 46)


def test_preprocessor_handles_unknown_categories():
    """Test that unseen categorical values do not cause errors."""

    df = load_data()
    X, _ = split_features_target(df)
    X = clean_features(X)

    X_train = X.iloc[:100].copy()
    X_new = X.iloc[[100]].copy()

    X_new.loc[:, "gender"] = "UnseenCategory"

    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_new)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == 46
