
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
data = pd.read_csv("data/train.csv")
X = data[["usage_hours", "monthly_charges"]]
y = data["churn"]

model = LogisticRegression()
model.fit(X, y)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
