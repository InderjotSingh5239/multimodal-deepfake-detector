import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
import time
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DeepGuard AI | Multimodal Forensics", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM THEMING (Professional Dark Mode) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; border: 1px solid #4x4x4x; }
    .stButton>button:hover { border: 1px solid #ff4b4b; color: #ff4b4b; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
    .report-box { padding: 20px; border: 1px solid #30363d; border-radius: 10px; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'history' not in st.session_state:
    st.session_state.history = []

# --- MOCK AUTHENTICATION SYSTEM ---
def login():
    st.sidebar.title("🔐 Secure Access")
    user = st.sidebar.text_input("Username", value="admin")
    pw = st.sidebar.text_input("Password", type="password", value="conference2024")
    if st.sidebar.button("Login"):
        if user == "admin" and pw == "conference2024":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.sidebar.error("Invalid Credentials")

# --- ANALYSIS ENGINE ---
def analyze_media(type="video"):
    progress = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress.progress(percent_complete + 1)
    
    # Mock analysis logic based on random patterns
    score = np.random.randint(40, 98)
    is_fake = score > 75
    return is_fake, score

# --- MAIN APP ROUTING ---
if not st.session_state.logged_in:
    st.title("🛡️ DeepGuard Multimodal Detection")
    st.info("Welcome to the DeepGuard Research Prototype. Please log in via the sidebar to access the forensic tools.")
    st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=1000")
    login()
else:
    # SIDEBAR NAVIGATION
    st.sidebar.title("🎮 Control Panel")
    menu = st.sidebar.radio("Navigation", ["📊 Dashboard", "📸 Live Camera", "🕵️ Face Detection", "📜 History"])
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # --- DASHBOARD PAGE ---
    if menu == "📊 Dashboard":
        st.title("📊 Forensic Dashboard")
        st.markdown("### Multimodal Analysis Upload")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload Media (Video/Audio)", type=['mp4', 'wav', 'mp3', 'mov'])
            if uploaded_file:
                st.video(uploaded_file)
                if st.button("Run Full Deepfake Audit"):
                    is_fake, score = analyze_media()
                    result = "FAKE" if is_fake else "REAL"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.history.append({"name": uploaded_file.name, "result": result, "score": score, "time": timestamp})
                    
                    with col2:
                        st.subheader("Audit Result")
                        if is_fake:
                            st.error(f"DETECTION: MANIPULATED CONTENT ({score}%)")
                        else:
                            st.success(f"DETECTION: AUTHENTIC CONTENT ({score}%)")
                        
                        # Gauage Chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = score,
                            title = {'text': "Authenticity Score"},
                            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#ff4b4b" if is_fake else "#00cc96"}}
                        ))
                        st.plotly_chart(fig, use_container_width=True)

    # --- LIVE CAMERA PAGE ---
    elif menu == "📸 Live Camera":
        st.title("📸 Real-time Stream Analysis")
        st.write("Capture a live stream to check for facial inconsistencies or synthetic overlays.")
        
        img_file = st.camera_input("Capture live face for scanning")
        
        if img_file:
            st.write("Frame Captured. Running Neural Scan...")
            is_fake, score = analyze_media()
            if is_fake:
                st.error(f"Potential Overlay Detected! Confidence: {score}%")
            else:
                st.success(f"Biometric Match Authentic. Confidence: {score}%")

    # --- FACE DETECTION PAGE ---
    elif menu == "🕵️ Face Detection":
        st.title("🕵️ Facial Feature Extraction")
        st.write("This module extracts spatial features and highlights anomalies in facial landmarks.")
        
        test_img = st.file_uploader("Upload image for landmark analysis", type=['jpg', 'png', 'jpeg'])
        if test_img:
            img = Image.open(test_img)
            img_array = np.array(img)
            
            # Simple simulation of landmark detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 5)
                cv2.putText(img_array, "Analyzing...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            st.image(img_array, caption="Scanning Spatio-Temporal Inconsistencies")
            st.info("Technical Note: The system is checking for 'Face-warping' and 'Moiré patterns' typical of AI re-enactment.")

    # --- HISTORY PAGE ---
    elif menu == "📜 History":
        st.title("📜 Forensic Audit History")
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                with st.expander(f"{item['time']} - {item['name']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Result:** {item['result']}")
                    c2.write(f"**Score:** {item['score']}%")
                    c3.write(f"**Status:** Archived")
        else:
            st.write("No forensic audits performed yet.")
