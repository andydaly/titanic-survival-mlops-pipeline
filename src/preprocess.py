import os
import pandas as pd


RAW_DATA_PATH = "data/Titanic-Dataset.csv"
PROCESSED_DATA_PATH = "data/processed_titanic.csv"


def preprocess_data():
    print("Starting Titanic data preprocessing...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    print("Original dataset shape:", df.shape)
    print("Original columns:", list(df.columns))

    # Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Keep useful columns only
    selected_columns = [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    df = df[selected_columns]

    # Fill missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["fare"] = df["fare"].fillna(df["fare"].median())

    # Convert categorical values to numbers
    df["sex"] = df["sex"].str.lower().map({
        "male": 0,
        "female": 1
    })

    df["embarked"] = df["embarked"].str.upper().map({
        "S": 0,
        "C": 1,
        "Q": 2
    })

    # Remove any remaining missing values
    df = df.dropna()

    # Make sure numeric columns are correct
    numeric_columns = [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    os.makedirs("data", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Processed dataset shape:", df.shape)
    print("Processed columns:", list(df.columns))
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_data()
