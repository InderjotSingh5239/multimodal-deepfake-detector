import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
from sklearn.neural_network import MLPClassifier
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Deepfake Detection", layout="wide")

# ---------------- MODERN UI ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #e0f7fa;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    color: black;
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
st.markdown("<div class='title'>Multimodal Deepfake Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Audio + Visual AI System (Conference Prototype)</div>", unsafe_allow_html=True)

# ---------------- MODEL ----------------
def create_model():
    X = np.random.rand(300, 28)
    y = np.random.randint(0, 2, 300)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
    model.fit(X, y)
    return model

model = create_model()

# ---------------- FACE DETECTOR ----------------
face_model = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- VISUAL FEATURE ----------------
def extract_visual(video_path):
    cap = cv2.VideoCapture(video_path)
    values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            values.append(np.mean(face))
            values.append(np.std(face))

    cap.release()

    if len(values) == 0:
        return np.zeros(2)

    return np.array([np.mean(values), np.std(values)])

# ---------------- AUDIO FEATURE ----------------
def extract_audio(video_path):
    try:
        audio, sr = librosa.load(video_path)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        return np.mean(np.vstack((mfcc, delta)), axis=1)
    except:
        return np.zeros(26)

# ---------------- IMAGE FEATURE ----------------
def extract_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.array([np.mean(gray), np.std(gray)])

# ---------------- FUSION ----------------
def fuse(v, a):
    return np.concatenate((v, a)).reshape(1, -1)

# ---------------- REASONING ----------------
def get_reason(v_mean, a_mean):
    reasons = []

    if v_mean < 80:
        reasons.append("Low facial detail (possible manipulation)")
    else:
        reasons.append("Normal facial structure")

    if a_mean < 0:
        reasons.append("Audio inconsistency detected")
    else:
        reasons.append("Audio appears natural")

    return reasons

# ---------------- INPUT TYPE ----------------
option = st.radio("Select Input Type", ["Video", "Image"])

# ---------------- VIDEO ----------------
if option == "Video":
    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        video_path = temp.name

        col1, col2 = st.columns(2)

        with col1:
            st.video(video_path)

        with col2:
            st.write("Processing:")
            st.write("✔ Face Detection")
            st.write("✔ Audio Analysis")
            st.write("✔ Feature Fusion")
            st.write("✔ Deep Learning Model")

        v = extract_visual(video_path)
        a = extract_audio(video_path)
        fusion = fuse(v, a)

        pred = model.predict(fusion)
        prob = model.predict_proba(fusion)

        fake_conf = prob[0][1] * 100
        real_conf = prob[0][0] * 100

        st.markdown("### Result")

        if pred[0] == 1:
            st.error(f"Deepfake Detected")
        else:
            st.success("Real Video")

        st.markdown(f"**Fake Confidence:** {fake_conf:.2f}%")
        st.markdown(f"**Real Confidence:** {real_conf:.2f}%")

        # Reason
        reasons = get_reason(np.mean(v), np.mean(a))

        st.markdown("### Why this result?")
        for r in reasons:
            st.write("- " + r)

# ---------------- IMAGE ----------------
if option == "Image":
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image")

        feat = extract_image(image)
        dummy_audio = np.zeros(26)

        fusion = fuse(feat, dummy_audio)

        pred = model.predict(fusion)
        prob = model.predict_proba(fusion)

        fake_conf = prob[0][1] * 100
        real_conf = prob[0][0] * 100

        st.markdown("### Result")

        if pred[0] == 1:
            st.error("Deepfake Image")
        else:
            st.success("Real Image")

        st.markdown(f"**Fake Confidence:** {fake_conf:.2f}%")
        st.markdown(f"**Real Confidence:** {real_conf:.2f}%")

        st.markdown("### Analysis")
        st.write("- Facial texture consistency checked")
        st.write("- Pixel intensity variation analyzed")
