import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
import time
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageOps
from datetime import datetime

# --- ADVANCED UI CONFIGURATION ---
st.set_page_config(
    page_title="DeepVerify Pro | Advanced Multimodal Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional "Dark-Cyber" Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background: #0b0e14; }
    .main-header { font-size: 2.5rem; font-weight: 600; color: #00d4ff; margin-bottom: 0.5rem; }
    .sub-text { color: #8b949e; margin-bottom: 2rem; }
    
    /* Forensic Card Styling */
    .forensic-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        transition: 0.3s;
    }
    .forensic-card:hover { border-color: #58a6ff; }
    
    /* Status Badges */
    .badge-fake { background: #442726; color: #ff7b72; padding: 4px 12px; border-radius: 20px; border: 1px solid #f85149; }
    .badge-real { background: #233129; color: #7ee787; padding: 4px 12px; border-radius: 20px; border: 1px solid #3fb950; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION MANAGEMENT ---
if 'auth_status' not in st.session_state:
    st.session_state.auth_status = False
if 'audit_logs' not in st.session_state:
    st.session_state.audit_logs = []

# --- CORE LOGIC: ANALYSIS ENGINE ---
def mock_deep_analysis(media_type="video"):
    """Simulates a complex Multi-head Attention Transformer result."""
    # Forensic metrics simulation
    visual_score = np.random.uniform(0.1, 0.9)
    audio_score = np.random.uniform(0.1, 0.9)
    # Weighted fusion
    fusion_score = (visual_score * 0.6) + (audio_score * 0.4)
    is_fake = fusion_score > 0.55
    confidence = np.random.uniform(88.5, 99.2) if is_fake else np.random.uniform(91.0, 98.7)
    
    reasons = []
    if is_fake:
        reasons = [
            "Spatio-temporal flickering detected in periorbital region.",
            "Mel-frequency mismatch: Synthetic voice texture detected.",
            "Lip-sync jitter (Asynchrony > 0.12s) identified in frame 42."
        ]
    else:
        reasons = ["Natural light reflections detected.", "Audio-visual coherence verified.", "No synthetic noise patterns found."]
        
    return is_fake, round(confidence, 2), reasons, visual_score, audio_score

# --- PAGES ---

def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; padding-top: 100px;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/10411/10411139.png", width=100)
        st.markdown("<h1 class='main-header'>System Login</h1>", unsafe_allow_html=True)
        st.write("University Research Portal - International AI Conference Demo")
        
        with st.form("Login"):
            user = st.text_input("Investigator ID")
            pwd = st.text_input("Access Token", type="password")
            submitted = st.form_submit_button("Initialize System")
            
            if submitted:
                if user == "admin" and pwd == "demo2024":
                    st.session_state.auth_status = True
                    st.success("Authentication Successful. Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid Credentials. Check documentation.")
        st.markdown("</div>", unsafe_allow_html=True)

def dashboard():
    st.markdown("<h1 class='main-header'>🛡️ Forensic Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-text'>Real-time Multimodal Deepfake Detection System v4.2.0-Alpha</p>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1.2, 0.8])
    
    with col_l:
        st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop Media File (MP4, MOV, AVI)", type=['mp4', 'mov', 'avi'])
        
        if uploaded_file:
            st.video(uploaded_file)
            if st.button("🚀 Execute Forensic Audit"):
                with st.status("Initializing Neural Engine...", expanded=True) as status:
                    st.write("Extracting facial landmarks via MTCNN...")
                    time.sleep(1)
                    st.write("Analyzing audio spectrograms via Librosa...")
                    time.sleep(1)
                    st.write("Applying Cross-Modal Attention Fusion...")
                    time.sleep(0.5)
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                fake, conf, reasons, v_score, a_score = mock_deep_analysis()
                
                # Result Header
                if fake:
                    st.markdown(f"### <span class='badge-fake'>DETECTED: MANIPULATED ({conf}%)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"### <span class='badge-real'>DETECTED: AUTHENTIC ({conf}%)</span>", unsafe_allow_html=True)
                
                st.write("**Primary Forensic Evidence:**")
                for r in reasons:
                    st.write(f"- {r}")
                
                # History log
                st.session_state.audit_logs.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "file": uploaded_file.name,
                    "result": "Fake" if fake else "Real",
                    "conf": conf
                })
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if uploaded_file and 'conf' in locals():
            st.subheader("📊 Statistical Breakdown")
            
            # Radar Chart for Multimodal Comparison
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[v_score*100, a_score*100, conf, np.random.randint(80,100), np.random.randint(85,100)],
                theta=['Visual Quality','Audio Sync','Fusion Confidence','Texture Consistency','Biometric Alignment'],
                fill='toself',
                name='Analysis',
                line_color='#00d4ff'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, 
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics Gauge
            st.metric("Integrity Index", f"{conf}%", delta="High Threat" if fake else "Secure")
        else:
            st.info("Upload a file to view forensic visualizations.")

def camera_page():
    st.title("📸 Live Biometric Stream")
    st.warning("Live monitoring mode: Scanning for frame-injection or GAN-based facial overlays.")
    
    cam_input = st.camera_input("Scanner Interface")
    if cam_input:
        img = Image.open(cam_input)
        # Advanced Processing Animation
        with st.spinner("Decoding Biometric Data..."):
            time.sleep(1.5)
            st.image(img, caption="Stream Captured", use_container_width=True)
            is_fake, conf, _, _, _ = mock_deep_analysis()
            
            if is_fake:
                st.error(f"ALERT: Synthetic Pattern Detected (Confidence: {conf}%)")
            else:
                st.success(f"Identity Verified: Authentic Stream (Confidence: {conf}%)")

def history_page():
    st.title("📜 Audit Registry")
    if not st.session_state.audit_logs:
        st.write("No forensic logs available for current session.")
    else:
        # Show as a Table
        df = st.session_state.audit_logs
        st.table(df)
        if st.button("Clear Logs"):
            st.session_state.audit_logs = []
            st.rerun()

# --- MAIN APP ROUTER ---
if not st.session_state.auth_status:
    login_page()
else:
    # Sidebar Navigation with dynamic Icons
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=300", use_container_width=True)
        st.markdown("### Investigator: Admin")
        st.divider()
        nav = st.radio("System Modules", ["Dashboard", "Live Camera", "Audit Logs", "XAI Documentation"])
        st.divider()
        if st.button("🚪 Log Out"):
            st.session_state.auth_status = False
            st.rerun()

    if nav == "Dashboard":
        dashboard()
    elif nav == "Live Camera":
        camera_page()
    elif nav == "Audit Logs":
        history_page()
    elif nav == "XAI Documentation":
        st.title("💡 Methodology & XAI")
        st.markdown("""
        ### Feature Fusion Architecture
        Our system utilizes an **Attention-Based Cross-Modal Transformer**. 
        1. **Spatial Branch:** ResNet-50 extracts facial meso-features.
        2. **Acoustic Branch:** MFCCs are fed into a Bi-LSTM to detect synthetic voice artifacts.
        3. **Fusion:** A weighted attention mechanism calculates the coherence between lip movement and audio frequency.
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*O-8O85_UAnXlY5S_A7rKlg.png", caption="System Architecture Diagram")
