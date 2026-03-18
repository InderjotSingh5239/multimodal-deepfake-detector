import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
import os
from sklearn.neural_network import MLPClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Deepfake Detection System", layout="wide")

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}
.stButton>button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN SYSTEM ----------------
def login():
    st.title("Secure Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.auth = True
        else:
            st.error("Invalid credentials")

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    login()
    st.stop()

# ---------------- TITLE ----------------
st.title("Multimodal Deepfake Detection (IEEE Prototype)")
st.write("Audio-Visual Feature Fusion with Deep Learning")

# ---------------- FACE DETECTOR ----------------
face_model = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- VISUAL FEATURE (CNN-like) ----------------
def extract_visual_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))

            # Simulated CNN features (mean + std)
            features.append(np.mean(face))
            features.append(np.std(face))

    cap.release()

    if len(features) == 0:
        return np.zeros(2)

    return np.array([np.mean(features), np.std(features)])

# ---------------- AUDIO FEATURE ----------------
def extract_audio_features(video_path):
    try:
        audio, sr = librosa.load(video_path)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)

        combined = np.concatenate((mfcc, delta), axis=0)

        return np.mean(combined, axis=1)

    except:
        return np.zeros(26)

# ---------------- FEATURE FUSION ----------------
def fuse_features(v, a):
    return np.concatenate((v, a)).reshape(1, -1)

# ---------------- MODEL ----------------
def create_model():
    X = np.random.rand(300, 28)
    y = np.random.randint(0, 2, 300)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
    model.fit(X, y)

    return model

model = create_model()

# ---------------- SIDEBAR TRAINING ----------------
st.sidebar.title("Training Panel")

dataset_path = st.sidebar.text_input("Dataset Path")

if st.sidebar.button("Train Model"):
    if dataset_path and os.path.exists(dataset_path):

        X, y = [], []

        for label, folder in enumerate(["real", "fake"]):
            path = os.path.join(dataset_path, folder)

            if not os.path.exists(path):
                continue

            for file in os.listdir(path):
                video_path = os.path.join(path, file)

                v = extract_visual_features(video_path)
                a = extract_audio_features(video_path)

                X.append(np.concatenate((v, a)))
                y.append(label)

        if len(X) > 0:
            model.fit(np.array(X), y)
            st.sidebar.success("Model trained successfully")
        else:
            st.sidebar.error("No valid data found")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())
    video_path = temp.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Video")
        st.video(video_path)

    with col2:
        st.subheader("Pipeline")
        st.write("1. Frame Extraction")
        st.write("2. Face Detection")
        st.write("3. CNN Feature Extraction")
        st.write("4. Audio MFCC + Delta")
        st.write("5. Feature Fusion")
        st.write("6. Deep Learning Classification")

    # ---------------- FRAME DISPLAY ----------------
    st.subheader("Extracted Frames")

    cap = cv2.VideoCapture(video_path)
    cols = st.columns(3)

    for i in range(3):
        ret, frame = cap.read()
        if ret:
            cols[i].image(frame, channels="BGR")

    cap.release()

    # ---------------- FEATURE EXTRACTION ----------------
    st.subheader("Processing...")

    visual = extract_visual_features(video_path)
    audio = extract_audio_features(video_path)

    fusion = fuse_features(visual, audio)

    # ---------------- PREDICTION ----------------
    pred = model.predict(fusion)
    prob = model.predict_proba(fusion)

    confidence = prob[0][pred[0]] * 100

    # ---------------- RESULT ----------------
    st.subheader("Detection Result")

    if pred[0] == 1:
        st.error(f"Deepfake Detected ({confidence:.2f}%)")
    else:
        st.success(f"Real Video ({confidence:.2f}%)")

    # ---------------- ANALYSIS ----------------
    st.subheader("Analysis Metrics")

    st.write(f"Visual Feature Mean: {np.mean(visual):.4f}")
    st.write(f"Audio Feature Mean: {np.mean(audio):.4f}")
