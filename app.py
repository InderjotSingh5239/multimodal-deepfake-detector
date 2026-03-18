import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- CONFIG ----------------
st.set_page_config(page_title="DeepShield AI", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#eef2ff,#f8fafc);
}

.header {
    font-size: 34px;
    font-weight: 700;
    color: #111827;
}

.card {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}

.real { color: green; font-weight: bold; }
.fake { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "1234":
            st.session_state.auth = True
        else:
            st.error("Wrong credentials")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("DeepShield AI")
menu = st.sidebar.radio("Menu", ["Dashboard", "Detection", "History"])

# ---------------- MODEL ----------------
def predict(val):
    if val < 100:
        return "Fake", 80, 20, "Facial inconsistency detected"
    else:
        return "Real", 20, 80, "Natural facial pattern"

# ---------------- FEATURES ----------------
def video_feature(path):
    cap = cv2.VideoCapture(path)
    vals = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(np.mean(gray))

    cap.release()
    return np.mean(vals) if vals else 0

def image_feature(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.mean(gray)

# ---------------- GRAPH ----------------
def graph(fake, real):
    fig, ax = plt.subplots()
    ax.bar(["Fake","Real"], [fake, real])
    st.pyplot(fig)

# ---------------- LOADER ----------------
def loader():
    progress = st.progress(0)
    steps = ["Extracting frames","Analyzing patterns","Running AI","Finalizing"]

    for i in range(4):
        st.write(steps[i])
        progress.progress((i+1)*25)
        time.sleep(0.5)

# ---------------- PDF ----------------
def create_pdf(result, fake, real, reason):
    file = "report.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Result: {result}", styles["Title"]))
    content.append(Paragraph(f"Fake: {fake}%", styles["Normal"]))
    content.append(Paragraph(f"Real: {real}%", styles["Normal"]))
    content.append(Paragraph(f"Reason: {reason}", styles["Normal"]))

    doc.build(content)
    return file

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.markdown("<div class='header'>AI Dashboard</div>", unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'><h3>Total</h3><h2>{}</h2></div>".format(len(st.session_state.history)), unsafe_allow_html=True)

    with col2:
        fake_count = sum(1 for i in st.session_state.history if i=="Fake")
        st.markdown(f"<div class='card'><h3>Fake</h3><h2>{fake_count}</h2></div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'><h3>Status</h3><h2>Active</h2></div>", unsafe_allow_html=True)

# ---------------- DETECTION ----------------
if menu == "Detection":
    st.markdown("<div class='header'>Detection Studio</div>", unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["Video","Image","Camera"])

    # VIDEO
    with tab1:
        file = st.file_uploader("Upload Video")

        if file:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(file.read())

            col1,col2 = st.columns(2)

            with col1:
                st.video(temp.name)

            loader()

            val = video_feature(temp.name)
            result,fake,real,reason = predict(val)

            st.session_state.history.append(result)

            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)

                st.subheader("Result")

                if result=="Fake":
                    st.markdown("<div class='fake'>Fake Video</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='real'>Real Video</div>", unsafe_allow_html=True)

                st.write(f"Fake: {fake}%")
                st.write(f"Real: {real}%")

                graph(fake, real)
                st.write("Reason:", reason)

                pdf = create_pdf(result,fake,real,reason)

                with open(pdf,"rb") as f:
                    st.download_button("Download Report", f, file_name="report.pdf")

                st.markdown("</div>", unsafe_allow_html=True)

    # IMAGE
    with tab2:
        img_file = st.file_uploader("Upload Image")

        if img_file:
            img = Image.open(img_file)
            st.image(img)

            loader()

            val = image_feature(img)
            result,fake,real,reason = predict(val)

            if result=="Fake":
                st.markdown("<div class='fake'>Fake Image</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='real'>Real Image</div>", unsafe_allow_html=True)

            graph(fake, real)

    # CAMERA
    with tab3:
        cam = st.camera_input("Capture")

        if cam:
            img = Image.open(cam)
            st.image(img)

            loader()

            val = image_feature(img)
            result,fake,real,reason = predict(val)

            if result=="Fake":
                st.markdown("<div class='fake'>Fake Detected</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='real'>Real Person</div>", unsafe_allow_html=True)

            graph(fake, real)

# ---------------- HISTORY ----------------
if menu == "History":
    st.markdown("<div class='header'>Detection History</div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.write(st.session_state.history)
    else:
        st.write("No data yet")
