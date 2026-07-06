import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


# 1. Charger le modèle
model = joblib.load("models/model.pkl")

# 2. Créer l'app FastAPI
app = FastAPI()


# 3. Définir le schéma d'entrée
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# 4. Endpoint de base
@app.get("/")
def home():
    return {"message": "ML Housing API is running"}


# 5. Endpoint de prédiction
@app.post("/predict")
def predict(features: HouseFeatures):
    data = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude
    ]])

    prediction = model.predict(data)[0]

    return {
        "prediction": float(prediction)
    }