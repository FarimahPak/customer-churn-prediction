
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"

RANDOM_STATE = 42

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

DECISION_THRESHOLD = 0.4

MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 2,
    "random_state": RANDOM_STATE,
}
