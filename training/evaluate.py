from pathlib import Path
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "validate.csv"
model_path = BASE_DIR / "model" / "model.pkl"

# ---------- Load data ----------
data = pd.read_csv(data_path)

print("Before cleaning:")
print(data)
print(data.isna().sum())

# ---------- Enforce numeric ----------
data["usage_hours"] = pd.to_numeric(data["usage_hours"], errors="coerce")
data["monthly_charges"] = pd.to_numeric(data["monthly_charges"], errors="coerce")

# ---------- Drop NaN ----------
data = data.dropna()

print("After cleaning:")
print(data)

if data.empty:
    raise Exception("No valid rows after cleaning validation data")

# ---------- Load model ----------
model = pickle.load(open(model_path, "rb"))

X = data[["usage_hours", "monthly_charges"]]
y = data["churn"]

acc = accuracy_score(y, model.predict(X))
print("Accuracy:", acc)

if acc < 0.75:
    raise Exception("Model quality gate failed")
