
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd

model = pickle.load(open("model/model.pkl", "rb"))
data = pd.read_csv("data/validate.csv")
print("DEBUG isna check:")
print(data.isna().sum())
``
X = data[["usage_hours", "monthly_charges"]]
y = data["churn"]

acc = accuracy_score(y, model.predict(X))
print("Accuracy:", acc)


if acc < 0.75:
    raise Exception("Model quality gate failed")
