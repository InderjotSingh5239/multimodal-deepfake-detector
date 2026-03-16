import numpy as np
from tensorflow.keras.models import load_model


model = load_model("models/deepfake_model.h5")


def predict(features):

    prediction = model.predict(features)

    if prediction > 0.5:
        return "Deepfake"
    else:
        return "Real"
