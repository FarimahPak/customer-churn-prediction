
import joblib

from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    MODEL_PARAMS,
)
from .data import load_data, split_features_target
from .preprocessing import clean_features, build_preprocessor


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
            (
                "cleaner",
                FunctionTransformer(clean_features, validate=False),
            ),
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                GradientBoostingClassifier(**MODEL_PARAMS)
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
