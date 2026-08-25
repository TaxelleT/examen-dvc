import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

PROCESSED_PATH = os.path.join("data", "processed_data")
MODELS_PATH = "models"
METRICS_PATH = "metrics"

X_test = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_PATH, "y_test.csv")).squeeze()

model = joblib.load(os.path.join(MODELS_PATH, "model.pkl"))

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

scores = {"mse": mse, "r2": r2}
os.makedirs(METRICS_PATH, exist_ok=True)
with open(os.path.join(METRICS_PATH, "scores.json"), "w") as f:
    json.dump(scores, f, indent=4)

predictions = pd.DataFrame({"y_test": y_test.values, "y_pred": y_pred})
predictions.to_csv(os.path.join(PROCESSED_PATH, "predictions.csv"), index=False)

print("Évaluation terminée.")
print("  → metrics/scores.json")
print("  → data/processed_data/predictions.csv")
