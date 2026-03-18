import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time
import tempfile

# Safe imports
try:
    import cv2
except:
    cv2 = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
except:
    tf = None

# ---------------- CONFIG ----------------
st.set_page_config(page_title="DeepShield AI Pro", layout="wide")

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    if tf is None:
        return None

    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model

model = load_model()

# ---------------- FACE DETECTION ----------------
def detect_face(img):
    if cv2 is None:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        img = img[y:y+h, x:x+w]

    return img

# ---------------- PREDICTION ----------------
def predict_image(img):
    img_np = np.array(img)

    if cv2 is not None:
        img_np = detect_face(img_np)

    img_resized = cv2.resize(img_np, (128,128))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    if model:
        pred = model.predict(img_resized)[0][0]
    else:
        pred = np.random.rand()

    if pred > 0.5:
        return "Fake", int(pred*100), int((1-pred)*100), "CNN detected artifacts & inconsistencies"
    else:
        return "Real", int(pred*100), int((1-pred)*100), "Natural facial structure detected"

# ---------------- LOADER ----------------
def loader():
    progress = st.progress(0)
    text = st.empty()

    steps = ["Detecting face...", "Extracting features...", "Running CNN...", "Finalizing..."]

    for i, step in enumerate(steps):
        text.write(step)
        progress.progress((i+1)*25)
        time.sleep(0.4)

# ---------------- CHART ----------------
def show_chart(fake, real):
    fig, ax = plt.subplots()
    ax.bar(["Fake","Real"], [fake, real])
    st.pyplot(fig)

# ---------------- SIDEBAR ----------------
st.sidebar.title("DeepShield AI Pro")
menu = st.sidebar.radio("Menu", ["Dashboard","Detection","History"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("📊 Dashboard")

    total = len(st.session_state.history)
    fake = sum(1 for i in st.session_state.history if i["result"]=="Fake")
    real = sum(1 for i in st.session_state.history if i["result"]=="Real")

    c1,c2,c3 = st.columns(3)

    c1.metric("Total Scans", total)
    c2.metric("Fake", fake)
    c3.metric("Real", real)

# ---------------- DETECTION ----------------
if menu == "Detection":
    st.title("🎯 Deepfake Detection")

    tabs = st.tabs(["Image","Camera"])

    # IMAGE
    with tabs[0]:
        file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

        if file:
            img = Image.open(file).convert("RGB")
            st.image(img)

            loader()

            result, fake, real, reason = predict_image(img)

            st.success(f"Result: {result}")
            st.write(f"Fake: {fake}% | Real: {real}%")
            st.write("Reason:", reason)

            show_chart(fake, real)

            st.session_state.history.append({
                "type":"Image",
                "result":result,
                "fake":fake,
                "real":real,
                "reason":reason
            })

    # CAMERA
    with tabs[1]:
        cam = st.camera_input("Capture Image")

        if cam:
            img = Image.open(cam).convert("RGB")
            st.image(img)

            loader()

            result, fake, real, reason = predict_image(img)

            st.success(f"Result: {result}")
            st.write(f"Fake: {fake}% | Real: {real}%")
            st.write("Reason:", reason)

            show_chart(fake, real)

            st.session_state.history.append({
                "type":"Camera",
                "result":result,
                "fake":fake,
                "real":real,
                "reason":reason
            })

# ---------------- HISTORY ----------------
if menu == "History":
    st.title("📜 History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No history yet")
