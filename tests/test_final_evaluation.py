
import numpy as np
import pandas as pd

from src.churn_prediction import final_evaluation


def test_final_evaluation_main(monkeypatch, capsys):
    fake_df = pd.DataFrame({
        "customerID": ["A", "B", "C", "D"],
        "tenure": [1, 2, 3, 4],
        "MonthlyCharges": [20, 30, 40, 50],
        "TotalCharges": [20, 60, 120, 200],
        "Churn": ["No", "Yes", "No", "Yes"],
    })

    fake_X = pd.DataFrame({"tenure": [1, 2, 3, 4]})
    fake_y = pd.Series([0, 1, 0, 1])

    fake_X_test = pd.DataFrame({"tenure": [4]})
    fake_y_test = pd.Series([1])

    class FakeModel:
        pass

    monkeypatch.setattr(
        final_evaluation,
        "load_data",
        lambda: fake_df,
    )

    monkeypatch.setattr(
        final_evaluation,
        "split_features_target",
        lambda df: (fake_X, fake_y),
    )

    monkeypatch.setattr(
        final_evaluation,
        "clean_features",
        lambda X: X,
    )

    monkeypatch.setattr(
        final_evaluation,
        "split_data",
        lambda X, y: (
            None,
            None,
            fake_X_test,
            None,
            None,
            fake_y_test,
        ),
    )

    monkeypatch.setattr(
        final_evaluation,
        "load_model",
        lambda: FakeModel(),
    )

    captured_evaluation = {}

    def fake_evaluate_model(model, X, y):
        captured_evaluation["model"] = model
        captured_evaluation["X"] = X
        captured_evaluation["y"] = y

        return (
            {"Accuracy": 0.9},
            np.array([[1, 0], [0, 1]]),
        )

    monkeypatch.setattr(
        final_evaluation,
        "evaluate_model",
        fake_evaluate_model,
    )

    monkeypatch.setattr(
        final_evaluation,
        "print_evaluation_results",
        lambda metrics, matrix, dataset_name: print(
            f"{dataset_name} evaluation completed"
        ),
    )

    final_evaluation.main()

    assert isinstance(captured_evaluation["model"], FakeModel)
    pd.testing.assert_frame_equal(
        captured_evaluation["X"],
        fake_X_test,
    )
    pd.testing.assert_series_equal(
        captured_evaluation["y"],
        fake_y_test,
    )

    captured = capsys.readouterr()
    assert "Test evaluation completed" in captured.out
