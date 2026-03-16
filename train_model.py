import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


X = np.random.rand(200, 14)
y = np.random.randint(0, 2, 200)


model = Sequential()

model.add(Dense(64, activation="relu", input_shape=(14,)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=10)

model.save("models/deepfake_model.h5")

print("Model saved successfully")
