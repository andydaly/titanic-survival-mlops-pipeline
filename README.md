# Titanic Survival MLOps Pipeline

Final MLOps Assignment

## Project Overview

This project implements an end-to-end MLOps pipeline for a Titanic survival prediction model.

The machine learning model predicts whether a Titanic passenger survived based on passenger details such as class, sex, age, number of siblings/spouses aboard, number of parents/children aboard, fare, and port of embarkation.

The project uses GitHub Actions to automate preprocessing, model training, testing, Docker image creation, Docker Hub publishing, and deployment to a Google VM using a self-hosted runner.

Dataset used:

https://www.kaggle.com/datasets/yasserh/titanic-dataset

---

## Use Case

The use case is a classification problem where the model predicts passenger survival.

Input example:

```json
{
  "pclass": 1,
  "sex": 1,
  "age": 30,
  "sibsp": 0,
  "parch": 0,
  "fare": 80,
  "embarked": 1
}
```

Output example:

```json
{
  "prediction": 1,
  "result": "Survived",
  "survival_probability": 1.0
}
```

Encoding used by the API:

```text
sex: male = 0, female = 1
embarked: S = 0, C = 1, Q = 2
```

---

## Project Structure

```text
titanic-survival-mlops-pipeline/
│
├── .github/workflows/
│   ├── ci.yml
│   ├── docker-publish.yml
│   └── deploy.yml
│
├── app/
│   └── flaskapp.py
│
├── data/
│   ├── Titanic-Dataset.csv
│   └── processed_titanic.csv
│
├── models/
│   ├── model.pkl
│   └── metrics.json
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── test_model.py
│
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## MLOps Stages Implemented

This project implements five MLOps stages.

### 1. Data Acquisition and Preprocessing

The raw Titanic dataset is stored in the `data/` directory.

The preprocessing script:

```text
src/preprocess.py
```

performs the following tasks:

- Loads the raw Titanic dataset.
- Selects useful model features.
- Handles missing values.
- Converts categorical values into numeric format.
- Saves the cleaned dataset as `data/processed_titanic.csv`.

Run locally:

```bash
python3 src/preprocess.py
```

---

### 2. Model Training and Testing

The model training script:

```text
src/train.py
```

trains a `RandomForestClassifier` to predict whether a passenger survived.

The script:

- Loads the processed dataset.
- Splits the data into training and testing sets.
- Trains a Random Forest classification model.
- Saves the trained model to `models/model.pkl`.
- Saves evaluation metrics to `models/metrics.json`.

Run locally:

```bash
python3 src/train.py
```

The testing script:

```text
src/test_model.py
```

checks that:

- The trained model file exists.
- The processed dataset exists.
- The model can generate a valid prediction.
- The model can return prediction probabilities.

Run tests:

```bash
python3 -m pytest src/test_model.py
```

---

### 3. Continuous Integration

Continuous Integration is implemented using GitHub Actions.

Workflow file:

```text
.github/workflows/ci.yml
```

The CI workflow runs automatically on pushes and pull requests to the `main` branch.

It performs the following steps:

- Checks out the repository.
- Installs Python dependencies.
- Runs data preprocessing.
- Trains the model.
- Runs automated model tests.
- Uploads model artefacts.

---

### 4. Model Deployment

The trained model is deployed through a Flask API.

Flask application:

```text
app/flaskapp.py
```

The API provides two endpoints:

```text
GET /
POST /predict
```

The `/predict` endpoint accepts passenger details in JSON format and returns a survival prediction.

Run locally:

```bash
python3 app/flaskapp.py
```

Test locally:

```bash
curl http://127.0.0.1:5000/
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "pclass": 1,
  "sex": 1,
  "age": 30,
  "sibsp": 0,
  "parch": 0,
  "fare": 80,
  "embarked": 1
}'
```

---

### 5. Continuous Delivery and Deployment

Continuous Delivery is implemented using GitHub Actions and Docker Hub.

Workflow file:

```text
.github/workflows/docker-publish.yml
```

This workflow:

- Runs preprocessing.
- Trains the model.
- Runs tests.
- Builds the Docker image.
- Pushes the image to Docker Hub.

Docker image:

```text
geniusatstuff/titanic-survival-api:latest
```

Automatic deployment is implemented using a self-hosted GitHub Actions runner on a Google VM.

Workflow file:

```text
.github/workflows/deploy.yml
```

This workflow:

- Pulls the latest Docker image from Docker Hub.
- Stops and removes the old container.
- Runs the latest container.
- Tests the deployed Flask API.

The deployed API runs on port `5000`.

---

## Docker Usage

Build the Docker image locally:

```bash
docker build -t titanic-survival-api:latest .
```

Run the local Docker container:

```bash
docker run -d -p 5000:5000 --name titanic-survival-api titanic-survival-api:latest
```

Pull the published image from Docker Hub:

```bash
docker pull geniusatstuff/titanic-survival-api:latest
```

Run the published image:

```bash
docker run -d -p 5000:5000 --name titanic-survival-api geniusatstuff/titanic-survival-api:latest
```

Stop and remove the container:

```bash
docker stop titanic-survival-api
docker rm titanic-survival-api
```

---

## API Usage

Home endpoint:

```bash
curl http://34.32.179.102:5000/
```

Prediction endpoint:

```bash
curl -X POST http://34.32.179.102:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "pclass": 1,
  "sex": 1,
  "age": 30,
  "sibsp": 0,
  "parch": 0,
  "fare": 80,
  "embarked": 1
}'
```

Example response:

```json
{
  "prediction": 1,
  "result": "Survived",
  "survival_probability": 1.0
}
```

---

## Branching Strategy

This project uses a simple GitHub Flow branching strategy.

The `main` branch represents the production-ready version of the project. Feature branches can be used for separate development tasks, such as:

```text
feature/preprocessing
feature/model-training
feature/flask-api
feature/docker
feature/deployment
```

Changes are developed on feature branches and merged into `main` through pull requests. When changes are pushed or merged into `main`, GitHub Actions automatically runs the CI/CD workflows.

This feature branch was used to demonstrate the branching strategy before merging changes back into main.

---

## Deployment Architecture

The deployment architecture is:

```text
GitHub Repository
      ↓
GitHub Actions CI
      ↓
Model training and testing
      ↓
Docker image build
      ↓
Docker Hub
      ↓
Self-hosted GitHub Actions runner on Google VM
      ↓
Docker container running Flask API
      ↓
External user sends prediction request
```

The application is deployed as a Docker container on a Google VM and exposed on port `5000`.

---

## Technologies Used

- Python
- pandas
- scikit-learn
- Flask
- pytest
- Docker
- Docker Hub
- GitHub Actions
- Google VM
- Self-hosted GitHub Actions runner

---

## References

- Kaggle Titanic Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset
- GitHub Actions Documentation: https://docs.github.com/en/actions
- Docker Documentation: https://docs.docker.com/
- Flask Documentation: https://flask.palletsprojects.com/
- scikit-learn Documentation: https://scikit-learn.org/

