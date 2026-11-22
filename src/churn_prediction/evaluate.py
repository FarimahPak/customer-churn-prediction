
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .config import DECISION_THRESHOLD


def evaluate_model(model, X_val, y_val, threshold=DECISION_THRESHOLD):
    """Evaluate a trained model on validation data."""

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1-score": f1_score(y_val, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_val, y_proba),
        "PR-AUC": average_precision_score(y_val, y_proba),
        "Brier Score": brier_score_loss(y_val, y_proba),
    }

    matrix = confusion_matrix(y_val, y_pred)

    return metrics, matrix


def print_evaluation_results(metrics, matrix, dataset_name="Validation"):
    print(f"{dataset_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)