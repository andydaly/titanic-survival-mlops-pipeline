import os
import joblib
import pandas as pd


MODEL_PATH = "models/model.pkl"
PROCESSED_DATA_PATH = "data/processed_titanic.csv"


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"


def test_processed_data_exists():
    assert os.path.exists(PROCESSED_DATA_PATH), (
        f"Processed data not found at {PROCESSED_DATA_PATH}"
    )


def test_model_can_predict():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(columns=["survived"])
    sample = X.head(1)

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_prediction_probability():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(columns=["survived"])
    sample = X.head(1)

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0][0] <= 1
    assert 0 <= probabilities[0][1] <= 1
