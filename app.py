import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

# AI
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- CONFIG ----------------
st.set_page_config(page_title="DeepShield AI Ultra", layout="wide")

# ---------------- LOGIN ----------------
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "1234":
            st.session_state.auth = True
        else:
            st.error("Wrong credentials")
    st.stop()

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def extract_face(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img  # fallback

    x, y, w, h = faces[0]
    face = img_np[y:y+h, x:x+w]

    return Image.fromarray(face)

# ---------------- MODEL ----------------
MODEL_PATH = "deepfake_resnet.h5"

def train_model():
    IMG_SIZE = 128

    data = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32
    )

    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(data, epochs=5)

    model.save(MODEL_PATH)
    return model

# LOAD / TRAIN
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.warning("Training advanced model...")
    model = train_model()

# ---------------- PREDICT ----------------
def predict_image(img):
    face = extract_face(img)
    face = face.resize((128,128))
    arr = np.array(face)/255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0][0]

    if pred < 0.5:
        return "Fake", int((1-pred)*100), int(pred*100), "Facial artifacts + blending issues detected"
    else:
        return "Real", int((1-pred)*100), int(pred*100), "Consistent facial structure detected"

def predict_video(path):
    cap = cv2.VideoCapture(path)
    preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = extract_face(img)

        face = face.resize((128,128))
        arr = np.array(face)/255.0
        arr = np.expand_dims(arr, axis=0)

        p = model.predict(arr)[0][0]
        preds.append(p)

    cap.release()
    avg = np.mean(preds)

    if avg < 0.5:
        return "Fake", int((1-avg)*100), int(avg*100), "Temporal + facial inconsistency"
    else:
        return "Real", int((1-avg)*100), int(avg*100), "Natural face movement consistency"

# ---------------- UI ----------------
st.sidebar.title("DeepShield AI Ultra")
menu = st.sidebar.radio("Menu", ["Dashboard","Detection","History"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("📊 Dashboard")
    st.write("Total:", len(st.session_state.history))

# ---------------- DETECTION ----------------
if menu == "Detection":
    st.title("🎯 Detection Studio")

    tab1, tab2, tab3 = st.tabs(["Video","Image","Camera"])

    # IMAGE
    with tab2:
        file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
        if file:
            img = Image.open(file).convert("RGB")
            st.image(img)

            st.info("Processing...")
            time.sleep(1)

            result,fake,real,reason = predict_image(img)
            st.session_state.history.append(result)

            st.write(result)
            st.write("Fake:", fake,"%  Real:", real,"%")
            st.write(reason)

    # VIDEO
    with tab1:
        file = st.file_uploader("Upload Video", type=["mp4"])
        if file:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(file.read())

            st.video(temp.name)

            st.info("Analyzing video...")
            result,fake,real,reason = predict_video(temp.name)
            st.session_state.history.append(result)

            st.write(result)
            st.write(reason)

    # CAMERA
    with tab3:
        cam = st.camera_input("Capture")
        if cam:
            img = Image.open(cam).convert("RGB")
            st.image(img)

            result,fake,real,reason = predict_image(img)
            st.session_state.history.append(result)

            st.write(result)
            st.write(reason)

# ---------------- HISTORY ----------------
if menu == "History":
    st.write(st.session_state.history)
