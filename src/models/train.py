import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

PROCESSED_PATH = os.path.join("data", "processed_data")
MODELS_PATH = "models"

X_train = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_PATH, "y_train.csv")).squeeze()

best_params = joblib.load(os.path.join(MODELS_PATH, "best_params.pkl"))

model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, os.path.join(MODELS_PATH, "model.pkl"))

print("Entraînement terminé. Modèle sauvegardé dans models/model.pkl")
