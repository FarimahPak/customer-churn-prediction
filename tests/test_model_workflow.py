
import numpy as np
import pytest

from src.churn_prediction.data import load_data, split_features_target
from src.churn_prediction.preprocessing import clean_features
from src.churn_prediction.train import (
    split_data,
    build_model,
    train_model,
)
from src.churn_prediction.evaluate import evaluate_model


@pytest.fixture(scope="module")
def data_splits():
    """Prepare the same data splits used in model development."""

    df = load_data()
    X, y = split_features_target(df)
    X = clean_features(X)

    return split_data(X, y)


@pytest.fixture(scope="module")
def trained_model(data_splits):
    """Train the final model once for workflow tests."""

    X_fit, X_val, X_test, y_fit, y_val, y_test = data_splits

    return train_model(X_fit, y_fit)


def test_split_data_sizes(data_splits):
    """Test the expected sizes of all data splits."""

    X_fit, X_val, X_test, y_fit, y_val, y_test = data_splits

    assert len(X_fit) == 4507
    assert len(X_val) == 1127
    assert len(X_test) == 1409


def test_data_splits_have_no_overlap(data_splits):
    """Test that training, validation, and test indexes are disjoint."""

    X_fit, X_val, X_test, _, _, _ = data_splits

    assert set(X_fit.index).isdisjoint(X_val.index)
    assert set(X_fit.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)


def test_build_model_configuration():
    """Test the final model configuration."""

    model = build_model()

    assert list(model.named_steps.keys()) == [
        "preprocessor",
        "classifier",
    ]

    classifier = model.named_steps["classifier"]

    assert classifier.n_estimators == 200
    assert classifier.learning_rate == 0.05
    assert classifier.max_depth == 2
    assert classifier.random_state == 42


def test_model_training_and_predictions(data_splits, trained_model):
    """Test that the model trains and produces valid probabilities."""

    X_fit, X_val, X_test, y_fit, y_val, y_test = data_splits

    probabilities = trained_model.predict_proba(X_val)

    assert probabilities.shape == (1127, 2)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_model_evaluation_matches_previous_results(
    data_splits,
    trained_model,
):
    """Test that validation results match the original notebook."""

    X_fit, X_val, X_test, y_fit, y_val, y_test = data_splits

    metrics, matrix = evaluate_model(
        trained_model,
        X_val,
        y_val,
    )

    assert matrix.tolist() == [
        [715, 113],
        [101, 198],
    ]

    assert metrics["Recall"] == pytest.approx(0.6622, abs=0.0001)
    assert metrics["F1-score"] == pytest.approx(0.6492, abs=0.0001)
    assert metrics["ROC-AUC"] == pytest.approx(0.8510, abs=0.0001)
    assert metrics["PR-AUC"] == pytest.approx(0.6707, abs=0.0001)
    assert metrics["Brier Score"] == pytest.approx(0.1328, abs=0.0001)



def test_main_runs_complete_training_workflow(monkeypatch, tmp_path, capsys):
    import pandas as pd

    from src.churn_prediction import train as train_module
    from src.churn_prediction import evaluate as evaluate_module

    fake_df = pd.DataFrame({
        "customerID": ["A", "B", "C", "D"],
        "tenure": [1, 2, 3, 4],
        "MonthlyCharges": [20, 30, 40, 50],
        "TotalCharges": [20, 60, 120, 200],
        "Churn": ["No", "Yes", "No", "Yes"],
    })

    fake_X = fake_df.drop(columns=["Churn", "customerID"])
    fake_y = pd.Series([0, 1, 0, 1])

    fake_X_fit = pd.DataFrame({"tenure": [1, 2]})
    fake_X_val = pd.DataFrame({"tenure": [3]})
    fake_X_test = pd.DataFrame({"tenure": [4]})

    fake_y_fit = pd.Series([0, 1])
    fake_y_val = pd.Series([0])
    fake_y_test = pd.Series([1])

    class FakeModel:
        pass

    saved = {}

    monkeypatch.setattr(train_module, "load_data", lambda: fake_df)
    monkeypatch.setattr(
        train_module,
        "split_features_target",
        lambda df: (fake_X, fake_y),
    )
    monkeypatch.setattr(train_module, "clean_features", lambda X: X)

    monkeypatch.setattr(
        train_module,
        "split_data",
        lambda X, y: (
            fake_X_fit,
            fake_X_val,
            fake_X_test,
            fake_y_fit,
            fake_y_val,
            fake_y_test,
        ),
    )

    monkeypatch.setattr(
        train_module,
        "train_model",
        lambda X, y: FakeModel(),
    )

    test_model_path = tmp_path / "models" / "final_model.joblib"
    monkeypatch.setattr(train_module, "MODEL_PATH", test_model_path)

    def fake_dump(model, path):
        saved["model"] = model
        saved["path"] = path

    monkeypatch.setattr(train_module.joblib, "dump", fake_dump)

    monkeypatch.setattr(
        evaluate_module,
        "evaluate_model",
        lambda model, X, y: (
            {"Accuracy": 0.9},
            np.array([[1, 0], [0, 1]]),
        ),
    )

    monkeypatch.setattr(
        evaluate_module,
        "print_evaluation_results",
        lambda metrics, matrix: print("Fake evaluation completed"),
    )

    train_module.main()

    assert isinstance(saved["model"], FakeModel)
    assert saved["path"] == test_model_path
    assert test_model_path.parent.exists()

    captured = capsys.readouterr()

    assert "Model training completed successfully!" in captured.out
    assert "Fake evaluation completed" in captured.out
