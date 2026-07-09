import os
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

load_dotenv()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Connexion PostgreSQL
DATABASE_URL = (
    f"postgresql://"
    f"{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)

engine = create_engine(DATABASE_URL)


def load_data():
    query = "SELECT * FROM housing"

    df = pd.read_sql(
        query,
        engine
    )

    return df


def train():

    # 1. Charger les données depuis PostgreSQL
    df = load_data()

    print(df.head())
    print("Dataset shape:", df.shape)


    # 2. Séparer features / target

    X = df.drop(columns=["Price"])
    y = df["Price"]


    # 3. Split train/test

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # 4. Modèle

    model = LinearRegression()


    # 5. Entraînement

    model.fit(
        X_train,
        y_train
    )


    # 6. Evaluation

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    print("RMSE:", rmse)


    # 7. Sauvegarde

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/model.pkl"
    )

    print("Model saved")


if __name__ == "__main__":
    train()