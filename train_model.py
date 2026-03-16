import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os


os.makedirs("models", exist_ok=True)


X = np.random.rand(200, 14)

y = np.random.randint(0, 2, 200)


model = LogisticRegression()

model.fit(X, y)


joblib.dump(model, "models/deepfake_model.pkl")

print("Model trained and saved successfully")
