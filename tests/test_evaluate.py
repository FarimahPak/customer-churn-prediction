
import numpy as np

from src.churn_prediction.evaluate import (
    evaluate_model,
    print_evaluation_results,
)


class DummyModel:
    def predict_proba(self, X):
        return np.array([
            [0.80, 0.20],
            [0.70, 0.30],
            [0.60, 0.40],
            [0.20, 0.80],
        ])


def test_evaluate_model_returns_expected_metrics():
    model = DummyModel()
    X_val = np.zeros((4, 1))
    y_val = np.array([0, 0, 1, 1])

    metrics, matrix = evaluate_model(
        model,
        X_val,
        y_val,
        threshold=0.4,
    )

    assert metrics["Accuracy"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["F1-score"] == 1.0
    assert matrix.tolist() == [[2, 0], [0, 2]]


def test_evaluate_model_handles_no_positive_predictions():
    model = DummyModel()
    X_val = np.zeros((4, 1))
    y_val = np.array([0, 0, 1, 1])

    metrics, matrix = evaluate_model(
        model,
        X_val,
        y_val,
        threshold=0.9,
    )

    assert metrics["Precision"] == 0.0
    assert metrics["Recall"] == 0.0
    assert metrics["F1-score"] == 0.0
    assert matrix.tolist() == [[2, 0], [2, 0]]


def test_print_evaluation_results(capsys):
    metrics = {
        "Accuracy": 0.85,
        "Precision": 0.75,
    }
    matrix = np.array([
        [10, 2],
        [3, 5],
    ])

    print_evaluation_results(metrics, matrix)

    captured = capsys.readouterr()

    assert "Validation Metrics:" in captured.out
    assert "Accuracy: 0.8500" in captured.out
    assert "Precision: 0.7500" in captured.out
    assert "Confusion Matrix:" in captured.out
    assert "[[10  2]" in captured.out
    assert "[ 3  5]]" in captured.out
