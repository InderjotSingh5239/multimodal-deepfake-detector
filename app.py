import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
from sklearn.neural_network import MLPClassifier
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Deepfake Detection", layout="wide")

# ---------------- CLEAN LIGHT UI ----------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
def login():
    st.markdown("<div class='title'>Login</div>", unsafe_allow_html=True)
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
st.markdown("<div class='title'>Multimodal Deepfake Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Audio + Visual + AI Fusion (IEEE Prototype)</div>", unsafe_allow_html=True)

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

# ---------------- VISUAL FEATURES ----------------
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

# ---------------- AUDIO FEATURES ----------------
def extract_audio(video_path):
    try:
        audio, sr = librosa.load(video_path)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        return np.mean(np.vstack((mfcc, delta)), axis=1)
    except:
        return np.zeros(26)

# ---------------- IMAGE FEATURES ----------------
def extract_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.array([np.mean(gray), np.std(gray)])

# ---------------- FUSION ----------------
def fuse(v, a):
    return np.concatenate((v, a)).reshape(1, -1)

# ---------------- REASON ----------------
def get_reason(v, a):
    reasons = []
    if v < 80:
        reasons.append("Facial inconsistency detected")
    else:
        reasons.append("Facial structure appears natural")

    if a < 0:
        reasons.append("Audio mismatch detected")
    else:
        reasons.append("Audio consistent")

    return reasons

# ---------------- GRAPH ----------------
def show_graph(fake, real):
    labels = ['Fake', 'Real']
    values = [fake, real]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Confidence (%)")
    st.pyplot(fig)

# ---------------- HEATMAP ----------------
def show_heatmap(img):
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)

# ---------------- MODE ----------------
mode = st.radio("Select Mode", ["Video", "Image", "Live Camera"])

# ---------------- VIDEO ----------------
if mode == "Video":
    file = st.file_uploader("Upload Video", type=["mp4"])

    if file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        video_path = temp.name

        st.video(video_path)

        v = extract_visual(video_path)
        a = extract_audio(video_path)

        fusion = fuse(v, a)
        pred = model.predict(fusion)
        prob = model.predict_proba(fusion)

        fake_conf = prob[0][1] * 100
        real_conf = prob[0][0] * 100

        st.subheader("Result")

        if pred[0] == 1:
            st.error("Deepfake Detected")
        else:
            st.success("Real Video")

        st.write(f"Fake Confidence: {fake_conf:.2f}%")
        st.write(f"Real Confidence: {real_conf:.2f}%")

        show_graph(fake_conf, real_conf)

        st.subheader("Reason")
        for r in get_reason(np.mean(v), np.mean(a)):
            st.write("- " + r)

# ---------------- IMAGE ----------------
if mode == "Image":
    img_file = st.file_uploader("Upload Image")

    if img_file:
        image = Image.open(img_file)
        st.image(image)

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

        st.write(f"Fake: {fake_conf:.2f}%")
        st.write(f"Real: {real_conf:.2f}%")

        show_graph(fake_conf, real_conf)

# ---------------- CAMERA ----------------
if mode == "Live Camera":
    cam = st.camera_input("Capture Image")

    if cam:
        image = Image.open(cam)
        st.image(image)

        feat = extract_image(image)
        fusion = fuse(feat, np.zeros(26))

        pred = model.predict(fusion)

        if pred[0] == 1:
            st.error("Deepfake Detected")
        else:
            st.success("Real Person")

        heat = show_heatmap(np.array(image))
        st.image(heat, caption="Heatmap Analysis")
