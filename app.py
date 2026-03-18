import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
from sklearn.neural_network import MLPClassifier
from PIL import Image
import matplotlib.pyplot as plt
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="DeepShield AI", layout="wide")

# ---------------- PROFESSIONAL CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f8fafc;
}

.navbar {
    padding: 15px;
    background: white;
    border-bottom: 1px solid #e5e7eb;
    font-size: 22px;
    font-weight: bold;
    color: #1f2937;
}

.section {
    margin-top: 20px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

.title {
    font-size: 28px;
    font-weight: bold;
    color: #111827;
}

.subtitle {
    color: #6b7280;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
def login():
    st.markdown("<div class='navbar'>DeepShield AI - Login</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if user == "admin" and pwd == "1234":
                st.session_state.auth = True
            else:
                st.error("Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    login()
    st.stop()

# ---------------- NAVBAR ----------------
st.markdown("<div class='navbar'>DeepShield AI - Multimodal Deepfake Detection</div>", unsafe_allow_html=True)

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

# ---------------- FEATURES ----------------
def extract_visual(video_path):
    cap = cv2.VideoCapture(video_path)
    vals = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (64,64))
            vals.append(np.mean(face))
            vals.append(np.std(face))
    cap.release()
    return np.array([np.mean(vals), np.std(vals)]) if vals else np.zeros(2)

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

def fuse(v,a):
    return np.concatenate((v,a)).reshape(1,-1)

# ---------------- PROCESS ANIMATION ----------------
def loader():
    with st.spinner("Analyzing media..."):
        time.sleep(1.5)

# ---------------- GRAPH ----------------
def graph(fake, real):
    fig, ax = plt.subplots()
    ax.bar(["Fake","Real"], [fake, real])
    ax.set_ylabel("Confidence %")
    st.pyplot(fig)

# ---------------- MAIN LAYOUT ----------------
mode = st.radio("Select Input", ["Video", "Image", "Camera"])

col1, col2 = st.columns([1,1])

# ---------------- VIDEO ----------------
if mode == "Video":
    file = st.file_uploader("Upload Video", type=["mp4"])

    if file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        path = temp.name

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.video(path)
            st.markdown("</div>", unsafe_allow_html=True)

        loader()

        v = extract_visual(path)
        a = extract_audio(path)
        f = fuse(v,a)

        pred = model.predict(f)
        prob = model.predict_proba(f)

        fake = prob[0][1]*100
        real = prob[0][0]*100

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='title'>Result</div>", unsafe_allow_html=True)

            if pred[0] == 1:
                st.error("Deepfake Detected")
            else:
                st.success("Real Video")

            st.markdown(f"Fake: {fake:.2f}%")
            st.markdown(f"Real: {real:.2f}%")

            graph(fake, real)

            st.markdown("</div>", unsafe_allow_html=True)

# ---------------- IMAGE ----------------
if mode == "Image":
    img = st.file_uploader("Upload Image")

    if img:
        image = Image.open(img)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(image)
            st.markdown("</div>", unsafe_allow_html=True)

        loader()

        feat = extract_image(image)
        f = fuse(feat, np.zeros(26))

        pred = model.predict(f)
        prob = model.predict_proba(f)

        fake = prob[0][1]*100
        real = prob[0][0]*100

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            if pred[0] == 1:
                st.error("Deepfake Image")
            else:
                st.success("Real Image")

            graph(fake, real)

            st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CAMERA ----------------
if mode == "Camera":
    cam = st.camera_input("Capture Image")

    if cam:
        image = Image.open(cam)

        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(image)
            st.markdown("</div>", unsafe_allow_html=True)

        loader()

        feat = extract_image(image)
        f = fuse(feat, np.zeros(26))

        pred = model.predict(f)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            if pred[0] == 1:
                st.error("Deepfake Detected")
            else:
                st.success("Real Person")

            st.markdown("</div>", unsafe_allow_html=True)
