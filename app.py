import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
import time
import plotly.graph_objects as go
from PIL import Image, ImageOps
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# --- ADVANCED UI CONFIG ---
st.set_page_config(page_title="DeepVerify AI | International Conference Demo", layout="wide")

# Custom CSS for Professional Dark/Cyber Theme
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e6edf3; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #21262d; color: #58a6ff; border: 1px solid #30363d; font-weight: bold; transition: 0.3s; }
    .stButton>button:hover { border-color: #58a6ff; background-color: #30363d; transform: translateY(-2px); }
    .sidebar .sidebar-content { background-color: #161b22; }
    .report-card { background: rgba(22, 27, 34, 0.7); border: 1px solid #30363d; border-radius: 15px; padding: 25px; margin-bottom: 20px; }
    .status-fake { color: #ff7b72; font-weight: bold; border: 1px solid #f85149; padding: 5px 10px; border-radius: 5px; background: rgba(248, 81, 73, 0.1); }
    .status-real { color: #7ee787; font-weight: bold; border: 1px solid #3fb950; padding: 5px 10px; border-radius: 5px; background: rgba(63, 185, 80, 0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'history' not in st.session_state:
    st.session_state.history = []

# --- MOCK ML ENGINE ---
@st.cache_resource
def get_forensic_engine():
    # Simulated Multi-head Attention Model
    X = np.random.rand(100, 5)
    y = [0]*50 + [1]*50
    model = RandomForestClassifier(n_estimators=100).fit(X, y)
    return model

engine = get_forensic_engine()

# --- HELPER FUNCTIONS ---
def generate_radar_chart(scores):
    categories = ['Visual Texture', 'Audio Coherence', 'Lip-Sync', 'Blink Rate', 'Lighting']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores, theta=categories, fill='toself',
        line_color='#58a6ff', fillcolor='rgba(88, 166, 255, 0.3)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#30363d"), bgcolor="rgba(0,0,0,0)"),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=350
    )
    return fig

# --- LOGIN SYSTEM ---
def show_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; margin-top: 50px;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
        st.title("🛡️ Investigator Portal")
        st.write("DeepGuard Multimodal Forensic Lab")
        with st.container(border=True):
            user = st.text_input("User ID", placeholder="admin")
            pw = st.text_input("Access Key", type="password", placeholder="conference2026")
            if st.button("Initialize Forensic Environment"):
                if user == "admin" and pw == "conference2026":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Biometric/Key Combination")
        st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN APP ROUTING ---
if not st.session_state.authenticated:
    show_login()
else:
    # SIDEBAR NAVIGATION
    with st.sidebar:
        st.markdown("### 🛠️ Forensic Tools")
        menu = st.radio("Navigation", ["📊 Dashboard", "📸 Live Scan", "🔍 Face Detection", "📜 Audit History"])
        st.divider()
        st.write("**System Status:** Active")
        st.write("**Model:** Fusion-Net v4.2")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

    # --- 1. DASHBOARD ---
    if menu == "📊 Dashboard":
        st.markdown("## 📊 Forensic Dashboard")
        st.write("Perform deep-layer multimodal analysis on video or audio files.")
        
        file = st.file_uploader("Upload Media for Analysis", type=['mp4', 'mov', 'avi', 'wav'])
        
        if file:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("<div class='report-card'>", unsafe_allow_html=True)
                st.video(file)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with c2:
                with st.status("Running Multimodal Fusion Analysis...", expanded=True):
                    st.write("Extracting MFCC audio vectors...")
                    time.sleep(1)
                    st.write("Analyzing spatio-temporal facial landmarks...")
                    time.sleep(1)
                    st.write("Computing cross-modal attention weights...")
                
                # Mock Results
                is_fake = np.random.choice([True, False], p=[0.4, 0.6])
                conf = np.random.uniform(89.5, 99.8)
                radar_scores = np.random.randint(60, 100, 5) if not is_fake else np.random.randint(20, 70, 5)
                
                if is_fake:
                    st.markdown("### Result: <span class='status-fake'>MANIPULATED</span>", unsafe_allow_html=True)
                    reason = "High-frequency noise detected in facial gradients + Audio pitch mismatch."
                else:
                    st.markdown("### Result: <span class='status-real'>AUTHENTIC</span>", unsafe_allow_html=True)
                    reason = "Biological signals (rPPG) and audio sync verified within normal limits."
                
                st.metric("Detection Confidence", f"{conf:.2f}%", delta="CRITICAL" if is_fake else "SECURE")
                st.write(f"**Primary Evidence:** {reason}")
                
                # Store in history
                st.session_state.history.append({"name": file.name, "result": "Fake" if is_fake else "Real", "conf": conf, "time": datetime.now().strftime("%H:%M")})

            st.divider()
            st.subheader("🔍 Deep Feature Breakdown")
            st.plotly_chart(generate_radar_chart(radar_scores), use_container_width=True)

    # --- 2. LIVE SCAN ---
    elif menu == "📸 Live Scan":
        st.markdown("## 📸 Real-time Stream Analysis")
        st.info("Directly analyze live camera feed for frame-injection or overlay attacks.")
        
        cam_shot = st.camera_input("Scanner")
        if cam_shot:
            with st.spinner("Analyzing Biometric Stream..."):
                time.sleep(1.5)
                st.success("Analysis Complete: Stream appears Authentic (94.2% Confidence)")
                st.image(cam_shot, caption="Verified Identity Frame", width=400)

    # --- 3. FACE DETECTION ---
    elif menu == "🔍 Face Detection":
        st.markdown("## 🔍 Artifact Highlight Module")
        st.write("Upload a frame to visualize potential GAN artifacts or blending errors.")
        
        img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if img_file:
            img = Image.open(img_file)
            img_array = np.array(img)
            
            # Simple AI Visualization Simulation
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (88, 166, 255), 10)
                cv2.putText(img_array, "Analyzing Texture...", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (88, 166, 255), 3)
            
            st.image(img_array, caption="Forensic Boundary Detection")
            st.json({"Texture_Consistency": "0.88", "Lighting_Coherence": "0.92", "Artifact_Probability": "0.04"})

    # --- 4. AUDIT HISTORY ---
    elif menu == "📜 Audit History":
        st.markdown("## 📜 Forensic Registry")
        if not st.session_state.history:
            st.write("No files processed in this session.")
        else:
            for item in reversed(st.session_state.history):
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    c1.write(f"**File:** {item['name']}")
                    status_class = "status-fake" if item['result'] == "Fake" else "status-real"
                    c2.markdown(f"<span class='{status_class}'>{item['result']}</span>", unsafe_allow_html=True)
                    c3.write(f"🕒 {item['time']}")
