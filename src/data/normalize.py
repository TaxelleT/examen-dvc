import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

PROCESSED_PATH = os.path.join("data", "processed_data")

X_train = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test.csv"))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train_scaled.to_csv(os.path.join(PROCESSED_PATH, "X_train_scaled.csv"), index=False)
X_test_scaled.to_csv(os.path.join(PROCESSED_PATH, "X_test_scaled.csv"), index=False)

joblib.dump(scaler, os.path.join("models", "scaler.pkl"))

print("Normalisation terminée.")
print(f"X_train_scaled : {X_train_scaled.shape}")
print(f"X_test_scaled  : {X_test_scaled.shape}")
