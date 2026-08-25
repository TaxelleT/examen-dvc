import pandas as pd
from sklearn.model_selection import train_test_split
import os

RAW_PATH = os.path.join("data", "raw_data", "raw.csv")
PROCESSED_PATH = os.path.join("data", "processed_data")

df = pd.read_csv(RAW_PATH)

df = df.drop(columns=["date"], errors="ignore")

X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

os.makedirs(PROCESSED_PATH, exist_ok=True)

X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_PATH, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_PATH, "y_test.csv"), index=False)

print(f"Split terminé : {len(X_train)} train / {len(X_test)} test")
