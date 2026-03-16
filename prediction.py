import joblib


model = joblib.load("models/deepfake_model.pkl")


def predict(features):

    prediction = model.predict(features)

    if prediction[0] == 1:
        return "Deepfake"
    else:
        return "Real"
