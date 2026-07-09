import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

from sklearn.datasets import fetch_california_housing
from sqlalchemy import create_engine


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


# Charger le dataset
data = fetch_california_housing()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["Price"] = data.target


print(df.head())


# Envoyer dans PostgreSQL
df.to_sql(
    "housing",
    engine,
    if_exists="replace",
    index=False
)


print("Data loaded successfully")