
import pandas as pd
import pytest

from src.churn_prediction.data import load_data, split_features_target


def test_load_data_success(tmp_path):
    """Test loading a valid dataset."""

    df = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "Churn": ["No", "Yes"],
        "tenure": [12, 24],
        "MonthlyCharges": [50.0, 80.0],
        "TotalCharges": [600.0, 1920.0],
    })

    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)

    loaded_df = load_data(file_path)

    assert loaded_df.shape == (2, 5)
    assert list(loaded_df.columns) == list(df.columns)


def test_load_data_file_not_found():
    """Test error when dataset does not exist."""

    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")


def test_load_data_missing_required_columns(tmp_path):
    """Test error when required columns are missing."""

    df = pd.DataFrame({
        "customerID": ["C001"],
        "Churn": ["No"],
    })

    file_path = tmp_path / "invalid_data.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_data(file_path)


def test_split_features_target_success():
    """Test separating features and encoding the target."""

    df = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "tenure": [12, 24],
        "Churn": ["No", "Yes"],
    })

    X, y = split_features_target(df)

    assert "Churn" not in X.columns
    assert y.tolist() == [0, 1]
    assert y.dtype == int


def test_split_features_target_invalid_values():
    """Test error when target contains unexpected values."""

    df = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "tenure": [12, 24],
        "Churn": ["No", "Unknown"],
    })

    with pytest.raises(ValueError, match="unexpected values"):
        split_features_target(df)


def test_split_features_target_missing_column():
    """Test error when target column is missing."""

    df = pd.DataFrame({
        "customerID": ["C001"],
        "tenure": [12],
    })

    with pytest.raises(ValueError, match="Target column"):
        split_features_target(df)
