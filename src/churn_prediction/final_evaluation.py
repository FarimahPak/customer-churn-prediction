
from .data import load_data, split_features_target
from .preprocessing import clean_features
from .train import split_data
from .predict import load_model
from .evaluate import evaluate_model, print_evaluation_results


def main():
    # Load and prepare the original dataset
    df = load_data()
    X, y = split_features_target(df)
    X = clean_features(X)

    # Recreate the exact same data split
    _, _, X_test, _, _, y_test = split_data(X, y)

    # Load the already trained model
    model = load_model()

    # Evaluate only on the untouched test set
    metrics, matrix = evaluate_model(model, X_test, y_test)

    print("\nFinal Test Set Evaluation")
    print_evaluation_results(metrics, matrix, dataset_name="Test")


if __name__ == "__main__":
    main()
