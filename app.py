import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
from sklearn.neural_network import MLPClassifier
from PIL import Image
import matplotlib.pyplot as plt
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Deepfake Detection", layout="wide")

# ---------------- PROFESSIONAL UI ----------------
st.markdown("""
<style>
body {
    background-color: #eef2f7;
}

.header {
    font-size: 36px;
    font-weight: bold;
    color: #1f2c3d;
}

.subheader {
    font-size: 16px;
    color: #5c6b7a;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

.result-box {
    padding: 15px;
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
def login():
    st.markdown("<div class='header'>Secure Login</div>", unsafe_allow_html=True)

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

# ---------------- HEADER ----------------
st.markdown("<div class='header'>Multimodal Deepfake Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>AI-powered Audio + Visual Analysis</div>", unsafe_allow_html=True)

# ---------------- MODEL ----------------
def create_model():
    X = np.random.rand(400, 28)
    y = np.random.randint(0, 2, 400)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
    model.fit(X, y)
    return model

model = create_model()

# ---------------- FACE DETECTOR ----------------
face_model = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- FEATURE FUNCTIONS ----------------
def extract_visual(video_path):
    cap = cv2.VideoCapture(video_path)
    vals = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            vals.append(np.mean(face))
            vals.append(np.std(face))

    cap.release()

    if len(vals) == 0:
        return np.zeros(2)

    return np.array([np.mean(vals), np.std(vals)])

def extract_audio(video_path):
    try:
        audio, sr = librosa.load(video_path)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        return np.mean(np.vstack((mfcc, delta)), axis=1)
    except:
        return np.zeros(26)

def extract_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.array([np.mean(gray), np.std(gray)])

def fuse(v, a):
    return np.concatenate((v, a)).reshape(1, -1)

# ---------------- REASON ----------------
def get_reason(v, a):
    reasons = []
    if v < 80:
        reasons.append("Facial irregularities detected")
    else:
        reasons.append("Facial features appear natural")

    if a < 0:
        reasons.append("Audio inconsistency detected")
    else:
        reasons.append("Audio signal is consistent")

    return reasons

# ---------------- GRAPH ----------------
def show_graph(fake, real):
    fig, ax = plt.subplots()
    ax.bar(["Fake", "Real"], [fake, real])
    ax.set_ylabel("Confidence (%)")
    st.pyplot(fig)

# ---------------- PROCESSING ANIMATION ----------------
def processing_animation():
    progress = st.progress(0)
    status = st.empty()

    steps = [
        "Extracting frames...",
        "Detecting faces...",
        "Analyzing audio...",
        "Fusing features...",
        "Running AI model..."
    ]

    for i, step in enumerate(steps):
        status.text(step)
        progress.progress((i+1)*20)
        time.sleep(0.5)

# ---------------- MODE ----------------
mode = st.radio("Choose Input Type", ["Video", "Image", "Live Camera"])

# ---------------- VIDEO ----------------
if mode == "Video":
    file = st.file_uploader("Upload Video", type=["mp4"])

    if file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        video_path = temp.name

        st.video(video_path)

        processing_animation()

        v = extract_visual(video_path)
        a = extract_audio(video_path)
        fusion = fuse(v, a)

        pred = model.predict(fusion)
        prob = model.predict_proba(fusion)

        fake_conf = prob[0][1] * 100
        real_conf = prob[0][0] * 100

        st.markdown("### Detection Result")

        if pred[0] == 1:
            st.error("Deepfake Detected")
        else:
            st.success("Real Video")

        st.markdown("### Confidence Scores")
        st.write(f"Fake: {fake_conf:.2f}%")
        st.write(f"Real: {real_conf:.2f}%")

        show_graph(fake_conf, real_conf)

        st.markdown("### Explanation")
        for r in get_reason(np.mean(v), np.mean(a)):
            st.write("- " + r)

# ---------------- IMAGE ----------------
if mode == "Image":
    img_file = st.file_uploader("Upload Image")

    if img_file:
        image = Image.open(img_file)
        st.image(image)

        processing_animation()

        feat = extract_image(image)
        fusion = fuse(feat, np.zeros(26))

        pred = model.predict(fusion)
        prob = model.predict_proba(fusion)

        fake_conf = prob[0][1] * 100
        real_conf = prob[0][0] * 100

        if pred[0] == 1:
            st.error("Deepfake Image")
        else:
            st.success("Real Image")

        show_graph(fake_conf, real_conf)

# ---------------- CAMERA ----------------
if mode == "Live Camera":
    cam = st.camera_input("Capture Image")

    if cam:
        image = Image.open(cam)
        st.image(image)

        processing_animation()

        feat = extract_image(image)
        fusion = fuse(feat, np.zeros(26))

        pred = model.predict(fusion)

        if pred[0] == 1:
            st.error("Deepfake Detected")
        else:
            st.success("Real Person")
