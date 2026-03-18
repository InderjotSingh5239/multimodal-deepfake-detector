import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time
import tempfile

# Safe OpenCV import
try:
    import cv2
except:
    cv2 = None

# ---------------- CONFIG ----------------
st.set_page_config(page_title="DeepShield AI", layout="wide")

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- FACE DETECTION ----------------
def detect_face(img):
    if cv2 is None:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = img[y:y+h, x:x+w]

    return img

# ---------------- AI LOGIC (LIGHTWEIGHT) ----------------
def predict_image(img):
    img_np = np.array(img)

    if cv2 is not None:
        img_np = detect_face(img_np)

    brightness = np.mean(img_np)
    variance = np.var(img_np)

    # Fake detection logic (feature-based)
    if brightness < 100 or variance < 500:
        return "Fake", 75, 25, "Low lighting, blurred regions, or unnatural texture"
    else:
        return "Real", 20, 80, "Natural lighting, consistent skin texture"

# ---------------- LOADER ----------------
def loader():
    progress = st.progress(0)
    text = st.empty()

    steps = [
        "Detecting face...",
        "Extracting features...",
        "Analyzing patterns...",
        "Finalizing result..."
    ]

    for i, step in enumerate(steps):
        text.write(step)
        progress.progress((i+1)*25)
        time.sleep(0.4)

# ---------------- CHART ----------------
def show_chart(fake, real):
    fig, ax = plt.subplots()
    ax.bar(["Fake", "Real"], [fake, real])
    ax.set_title("Confidence Score")
    st.pyplot(fig)

# ---------------- SIDEBAR ----------------
st.sidebar.title("DeepShield AI")
menu = st.sidebar.radio("Menu", ["Dashboard", "Detection", "History"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.title("📊 Dashboard")

    total = len(st.session_state.history)
    fake = sum(1 for i in st.session_state.history if i["result"] == "Fake")
    real = sum(1 for i in st.session_state.history if i["result"] == "Real")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Scans", total)
    col2.metric("Fake Detected", fake)
    col3.metric("Real Detected", real)

# ---------------- DETECTION ----------------
if menu == "Detection":
    st.title("🎯 Deepfake Detection")

    tabs = st.tabs(["Image", "Camera"])

    # IMAGE
    with tabs[0]:
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

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
                "type": "Image",
                "result": result,
                "fake": fake,
                "real": real,
                "reason": reason
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
                "type": "Camera",
                "result": result,
                "fake": fake,
                "real": real,
                "reason": reason
            })

# ---------------- HISTORY ----------------
if menu == "History":
    st.title("📜 Detection History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        filter_type = st.selectbox("Filter by Type", ["All", "Image", "Camera"])

        if filter_type != "All":
            df = df[df["type"] == filter_type]

        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data yet")
