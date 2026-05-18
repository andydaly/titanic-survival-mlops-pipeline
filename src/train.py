import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


PROCESSED_DATA_PATH = "data/processed_titanic.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"


def train_model():
    print("Starting Titanic survival model training...")

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_DATA_PATH}. "
            "Run src/preprocess.py first."
        )

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)

    target_column = "survived"

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "model_type": "RandomForestClassifier",
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1_score": round(f1_score(y_test, predictions), 4),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features": list(X.columns)
    }

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    train_model()
