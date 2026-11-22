
import pandas as pd
import pytest

from churn_prediction.preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    clean_features,
    build_preprocessor,
    validate_customer_data,
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

def make_valid_customer_data():
    """Create synthetic customer data with valid raw input values."""

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
        "PaymentMethod": ["Electronic check", "Mailed check"],
    })

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

def test_validate_customer_data_accepts_valid_input():
    """Test that valid customer data passes validation."""

    X = make_valid_customer_data()

    validated = validate_customer_data(X)

    assert validated.shape == X.shape
    assert validated["customerID"].tolist() == ["C001", "C002"]
    assert validated["TotalCharges"].tolist() == [600.0, 0.0]
    assert pd.api.types.is_numeric_dtype(validated["TotalCharges"])
    assert validated["TotalCharges"].isna().sum() == 0

def test_validate_customer_data_rejects_missing_columns():
    """Test that missing required columns raise an error."""

    X = make_valid_customer_data().drop(columns=["tenure"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_customer_data(X)

def test_validate_customer_data_rejects_extra_columns():
    """Test that unexpected columns raise an error."""

    X = make_valid_customer_data()
    X["UnexpectedColumn"] = "value"

    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_customer_data(X)

def test_validate_customer_data_rejects_invalid_categories():
    """Test that invalid categorical values raise an error."""

    X = make_valid_customer_data()
    X.loc[0, "Contract"] = "Invalid Contract"

    with pytest.raises(ValueError, match="Invalid values in 'Contract'"):
        validate_customer_data(X)


def test_validate_customer_data_rejects_invalid_customer_ids():
    """Test that missing or empty customer IDs raise an error."""

    X = make_valid_customer_data()
    X.loc[0, "customerID"] = " "

    with pytest.raises(ValueError, match="customerID"):
        validate_customer_data(X)

def test_validate_customer_data_rejects_non_numeric_values():
    """Test that non-numeric values in numerical columns are rejected."""

    X = make_valid_customer_data()
    X["MonthlyCharges"] = X["MonthlyCharges"].astype(object)
    X.loc[0, "MonthlyCharges"] = "invalid"

    with pytest.raises(ValueError, match="non-numeric"):
        validate_customer_data(X)

def test_validate_customer_data_rejects_values_outside_allowed_ranges():
    """Test that numerical values outside allowed ranges are rejected."""

    X = make_valid_customer_data()
    X.loc[0, "tenure"] = 100

    with pytest.raises(ValueError, match="between"):
        validate_customer_data(X)

def test_validate_customer_data_rejects_fractional_tenure():
    """Test that tenure must contain integer values."""

    X = make_valid_customer_data()
    X["tenure"] = X["tenure"].astype(float)
    X.loc[0, "tenure"] = 12.5

    with pytest.raises(ValueError, match="integer"):
        validate_customer_data(X)

def test_validate_customer_data_rejects_empty_dataframe():
    """Test that empty customer data is rejected."""

    X = make_valid_customer_data().iloc[0:0]

    with pytest.raises(ValueError, match="at least one row"):
        validate_customer_data(X)


def test_validate_customer_data_rejects_non_dataframe_input():
    """Test that non-DataFrame input is rejected."""

    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_customer_data([{"tenure": 12}])


def test_validate_customer_data_rejects_duplicate_columns():
    """Test that duplicate column names are rejected."""

    X = make_valid_customer_data()
    X.columns = [
        "tenure" if column == "MonthlyCharges" else column
        for column in X.columns
    ]

    with pytest.raises(ValueError, match="Duplicate columns"):
        validate_customer_data(X)