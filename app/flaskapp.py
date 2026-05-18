import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run src/train.py first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Titanic Survival Prediction API is running",
        "usage": "Send a POST request to /predict with passenger details"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "embarked"
        ]

        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        input_data = pd.DataFrame([{
            "pclass": data["pclass"],
            "sex": data["sex"],
            "age": data["age"],
            "sibsp": data["sibsp"],
            "parch": data["parch"],
            "fare": data["fare"],
            "embarked": data["embarked"]
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = "Survived" if prediction == 1 else "Did not survive"

        return jsonify({
            "prediction": int(prediction),
            "result": result,
            "survival_probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
