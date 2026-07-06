import os
import joblib
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 1. Charger les données
data = fetch_california_housing()
X = data.data
y = data.target

print("Shape X:", X.shape)
print("Shape y:", y.shape)


# 2. Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Créer le modèle
model = LinearRegression()

# 4. Entraîner
model.fit(X_train, y_train)

# 5. Prédire
y_pred = model.predict(X_test)

# 6. Évaluer
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE:", rmse)


# 7. Sauvegarder le modèle
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")

print("Model saved in models/model.pkl")