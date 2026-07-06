import joblib
import numpy as np

# 1. Charger le modèle entraîné
model = joblib.load("models/model.pkl")

print("Model loaded successfully")

# 2. Exemple de données (1 maison)
# ordre des features = dataset California Housing
sample = np.array([[
    8.3252,   # MedInc
    41.0,     # HouseAge
    6.984127, # AveRooms
    1.023810, # AveBedrms
    322.0,    # Population
    2.555556, # AveOccup
    37.88,    # Latitude
    -122.23   # Longitude
]])

# 3. Prédiction
prediction = model.predict(sample)

print("Predicted house value:", prediction[0])