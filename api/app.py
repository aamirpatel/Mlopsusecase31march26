
from fastapi import FastAPI
import pickle

app = FastAPI()
model = pickle.load(open("model/model.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    X = [[data["usage_hours"], data["monthly_charges"]]]
    return {"churn": int(model.predict(X)[0])}
