import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

PROCESSED_PATH = os.path.join("data", "processed_data")
MODELS_PATH = "models"

X_train = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_PATH, "y_train.csv")).squeeze()

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Meilleurs paramètres : {best_params}")

os.makedirs(MODELS_PATH, exist_ok=True)
joblib.dump(best_params, os.path.join(MODELS_PATH, "best_params.pkl"))

print("GridSearch terminé. Paramètres sauvegardés dans models/best_params.pkl")
