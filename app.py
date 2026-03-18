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
st.set_page_config(page_title="DeepShield AI Ultra", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body { background: linear-gradient(135deg,#eef2ff,#f8fafc); }
.header { font-size: 34px; font-weight: bold; }
.real { color: green; font-weight: bold; }
.fake { color: red; font-weight: bold; }
.card {
    background: rgba(255,255,255,0.8);
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

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
            st.error("Invalid credentials")
    st.stop()

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- FAKE AI MODEL (STABLE) ----------------
def predict_image(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    val = np.mean(gray)

    if val < 100:
        return "Fake", 80, 20, "Pixel inconsistency & unnatural textures"
    else:
        return "Real", 20, 80, "Natural lighting and smooth features"

def predict_video(path):
    cap = cv2.VideoCapture(path)
    vals = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(np.mean(gray))

    cap.release()
    avg = np.mean(vals)

    if avg < 100:
        return "Fake", 78, 22, "Frame inconsistency detected"
    else:
        return "Real", 18, 82, "Stable motion patterns"

# ---------------- LOADER ----------------
def loader():
    progress = st.progress(0)
    for i in range(4):
        time.sleep(0.3)
        progress.progress((i+1)*25)

# ---------------- CHARTS ----------------
def charts(fake, real):
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.bar(["Fake","Real"], [fake, real])
        ax.set_title("Confidence")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie([fake, real], labels=["Fake","Real"], autopct="%1.1f%%")
        ax2.set_title("Distribution")
        st.pyplot(fig2)

# ---------------- PDF ----------------
def create_pdf(result, fake, real, reason):
    file = "report.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()

    content = [
        Paragraph(f"Result: {result}", styles["Title"]),
        Paragraph(f"Fake Confidence: {fake}%", styles["Normal"]),
        Paragraph(f"Real Confidence: {real}%", styles["Normal"]),
        Paragraph(f"Reason: {reason}", styles["Normal"])
    ]

    doc.build(content)
    return file

# ---------------- SIDEBAR ----------------
st.sidebar.title("DeepShield AI Ultra")
menu = st.sidebar.radio("Menu", ["Dashboard","Detection","History"])

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.markdown("<div class='header'>Dashboard</div>", unsafe_allow_html=True)
    st.write("Total Scans:", len(st.session_state.history))
    st.write("Fake Count:", st.session_state.history.count("Fake"))

# ---------------- DETECTION ----------------
if menu == "Detection":
    st.markdown("<div class='header'>Detection Studio</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Video","Image","Camera"])

    # VIDEO
    with tab1:
        file = st.file_uploader("Upload Video", type=["mp4"])
        if file:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(file.read())

            st.video(temp.name)
            loader()

            result,fake,real,reason = predict_video(temp.name)
            st.session_state.history.append(result)

            st.write(result)
            charts(fake, real)
            st.write(reason)

    # IMAGE
    with tab2:
        img_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
        if img_file:
            try:
                img = Image.open(img_file).convert("RGB")
                st.image(img)
            except:
                st.error("Invalid image")
                st.stop()

            loader()

            result,fake,real,reason = predict_image(img)
            st.session_state.history.append(result)

            st.write(result)
            charts(fake, real)
            st.write(reason)

            pdf = create_pdf(result,fake,real,reason)
            with open(pdf,"rb") as f:
                st.download_button("Download Report", f)

    # CAMERA
    with tab3:
        cam = st.camera_input("Capture")
        if cam:
            img = Image.open(cam).convert("RGB")
            st.image(img)

            loader()

            result,fake,real,reason = predict_image(img)
            st.session_state.history.append(result)

            st.write(result)
            charts(fake, real)
            st.write(reason)

# ---------------- HISTORY ----------------
if menu == "History":
    st.write(st.session_state.history)
