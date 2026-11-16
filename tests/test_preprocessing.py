
import pandas as pd

from src.churn_prediction.preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    clean_features,
    build_preprocessor,
)


def make_customer_data():
    """Create synthetic customer data with all expected features."""

    data = {
        "tenure": [12, 24, 36],
        "MonthlyCharges": [50.0, 80.0, 100.0],
        "TotalCharges": [600.0, 1920.0, 3600.0],
    }

    for feature in CATEGORICAL_FEATURES:
        data[feature] = ["Category_A", "Category_B", "Category_A"]

    return pd.DataFrame(data)


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
    """Test the transformed feature shape using synthetic data."""

    X = make_customer_data()

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    expected_features = len(NUMERICAL_FEATURES) + 2 * len(CATEGORICAL_FEATURES)

    assert X_transformed.shape == (3, expected_features)


def test_preprocessor_handles_unknown_categories():
    """Test that unseen categorical values do not cause errors."""

    X_train = make_customer_data()
    X_new = X_train.iloc[[0]].copy()

    X_new.loc[:, "gender"] = "UnseenCategory"

    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_new)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == (
        len(NUMERICAL_FEATURES) + 2 * len(CATEGORICAL_FEATURES)
    )
