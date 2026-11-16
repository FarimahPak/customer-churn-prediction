
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import load_data, split_features_target
from .preprocessing import clean_features, build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def split_data(X, y):
    """Split data into training, validation, and test sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    return X_fit, X_val, X_test, y_fit, y_val, y_test


def build_model() -> Pipeline:
    """Build the final Gradient Boosting model pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def train_model(X_fit, y_fit) -> Pipeline:
    """Train the model on the training data."""

    model = build_model()
    model.fit(X_fit, y_fit)

    return model



def main():
    df = load_data()
    X, y = split_features_target(df)

    X = clean_features(X)

    X_fit, X_val, X_test, y_fit, y_val, y_test = split_data(X, y)

    model = train_model(X_fit, y_fit)
    

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

    print("Model training completed successfully!")
    print("Training samples:", len(X_fit))
    print("Validation samples:", len(X_val))
    print("Test samples:", len(X_test))

    from .evaluate import evaluate_model, print_evaluation_results

    metrics, matrix = evaluate_model(model, X_val, y_val)

    print("\nFinal Model Evaluation")
    print_evaluation_results(metrics, matrix)


if __name__ == "__main__":
    main()
