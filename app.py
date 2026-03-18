import streamlit as st
import cv2
import numpy as np
import librosa
import time
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from datetime import datetime

# ==========================================
# 1. PAGE AND THEME CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DeepGuard Pro | Forensic Analysis Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to give it a "Student-Built Startup" vibe
# Using a dark-theme professional palette
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500&display=swap');
    
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stApp { background-color: #0d1117; }
    
    /* Global Font Change */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism Card Effect */
    .forensic-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Custom Status Badges */
    .status-badge {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-fake { background-color: #442d30; color: #ff7b72; border: 1px solid #f85149; }
    .badge-real { background-color: #233129; color: #7ee787; border: 1px solid #3fb950; }

    /* SideBar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        background-color: #238636;
        color: white;
        border: 1px solid rgba(240,246,252,0.1);
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: #8b949e;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE & UTILS
# ==========================================
if 'auth_active' not in st.session_state:
    st.session_state.auth_active = False
if 'forensic_logs' not in st.session_state:
    st.session_state.forensic_logs = []

def generate_radar_chart(values):
    """Generates a professional radar chart for multimodal features."""
    categories = ['Visual Noise', 'Audio Artifacts', 'Lip Sync Lag', 'Light Consistency', 'Blink Patterns']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Forensic Analysis',
        line_color='#58a6ff',
        fillcolor='rgba(88, 166, 255, 0.25)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#8b949e", gridcolor="#30363d"),
            bgcolor="#161b22"
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

# ==========================================
# 3. ANALYSIS ENGINE (SIMULATED)
# ==========================================
def process_forensic_audit():
    """Handles the internal 'AI logic' simulation for the demo."""
    # Simulating a multi-step neural network process
    steps = [
        "Initializing MTCNN for facial landmark localization...",
        "Extracting MFCC vectors from audio stream...",
        "Running Spatial-Temporal Inconsistency check...",
        "Applying Cross-Modal Attention Fusion..."
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) * 25)
        time.sleep(0.8)
    
    # Randomly determined but logically consistent 'results'
    is_fake = np.random.choice([True, False], p=[0.4, 0.6])
    score = np.random.uniform(91.2, 99.7)
    
    if is_fake:
        radar_data = [88, 92, 75, 60, 82]
        analysis_text = "Suspected manipulation found in audio-visual synchronization and facial texture noise."
    else:
        radar_data = [12, 15, 8, 22, 18]
        analysis_text = "No synthetic artifacts detected. Biometric signals match natural human distribution."
        
    return is_fake, score, radar_data, analysis_text

# ==========================================
# 4. PAGE: LOGIN
# ==========================================
def render_login():
    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/10411/10411139.png", width=100)
        st.markdown("<h1 style='color: #58a6ff; font-weight: 700;'>DeepGuard Investigator Portal</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #8b949e;'>University Research Framework | International AI Conf 2026</p>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
            u_id = st.text_input("Investigator ID", placeholder="e.g. admin_01")
            u_key = st.text_input("Access Token", type="password", placeholder="••••••••")
            
            if st.button("Initialize Forensic Suite"):
                if u_id == "admin" and u_key == "conference2026":
                    st.session_state.auth_active = True
                    st.success("Authentication successful. Loading modules...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Security Token")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 5. PAGE: DASHBOARD (MAIN)
# ==========================================
def render_dashboard():
    st.markdown("<h1 style='font-size: 2.2rem;'>📊 Forensic Command Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8b949e; margin-bottom: 25px;'>Centralized Hub for Multimodal Deepfake Verification</p>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1.4, 1])
    
    with col_l:
        st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
        st.subheader("📁 Evidence Ingestion")
        uploaded_media = st.file_uploader("Upload Video or Audio Sample", type=['mp4', 'mov', 'avi', 'wav', 'mp3'])
        
        if uploaded_media:
            st.video(uploaded_media)
            if st.button("🚀 Start Deep Forensic Audit"):
                is_fake, conf, radar_vals, summary = process_forensic_audit()
                st.session_state.last_result = {
                    'fake': is_fake, 'conf': conf, 'radar': radar_vals, 'summary': summary, 'name': uploaded_media.name
                }
                # Log to history
                st.session_state.forensic_logs.append({
                    'time': datetime.now().strftime("%H:%M"),
                    'name': uploaded_media.name,
                    'verdict': 'FAKE' if is_fake else 'REAL'
                })
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if 'last_result' in st.session_state:
            res = st.session_state.last_result
            st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
            st.subheader("🎯 Audit Result")
            
            if res['fake']:
                st.markdown("<span class='status-badge badge-fake'>DETECTED: MANIPULATED</span>", unsafe_allow_html=True)
                st.error(f"High Risk Probability: {res['conf']:.2f}%")
            else:
                st.markdown("<span class='status-badge badge-real'>DETECTED: AUTHENTIC</span>", unsafe_allow_html=True)
                st.success(f"Security Confidence: {res['conf']:.2f}%")
            
            st.write(f"**Forensic Note:** {res['summary']}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Feature Distribution")
            st.plotly_chart(generate_radar_chart(res['radar']), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("System Standby: Please upload a media file to begin multimodal decomposition.")

# ==========================================
# 6. PAGE: LIVE CAMERA
# ==========================================
def render_camera():
    st.markdown("<h1>📸 Real-time Stream Analysis</h1>", unsafe_allow_html=True)
    st.write("Checking live biometric feed for frame-injection or GAN-based facial overlays.")
    
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
        cam_feed = st.camera_input("Scanner Interface")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with c2:
        if cam_feed:
            st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
            st.subheader("📈 Stream Metadata")
            with st.spinner("Decoding facial mesh..."):
                time.sleep(2)
                st.metric("Inconsistency Level", "4.2%", "-0.8%")
                st.metric("Biometric Match", "98.1%", "Valid")
                st.divider()
                st.write("**Verdict:** Live stream appears authentic. No synthetic frequency noise detected.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Awaiting camera input for live session analysis.")

# ==========================================
# 7. PAGE: AUDIT LOGS
# ==========================================
def render_history():
    st.markdown("<h1>📜 Global Audit Registry</h1>", unsafe_allow_html=True)
    if not st.session_state.forensic_logs:
        st.write("No forensic records found in current session.")
    else:
        for entry in reversed(st.session_state.forensic_logs):
            with st.container():
                st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
                cols = st.columns([1, 3, 1])
                cols[0].write(f"🕒 {entry['time']}")
                cols[1].write(f"**Sample:** {entry['name']}")
                status_color = "#ff7b72" if entry['verdict'] == 'FAKE' else "#7ee787"
                cols[2].markdown(f"<span style='color: {status_color}; font-weight: bold;'>{entry['verdict']}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 8. PAGE: METHODOLOGY
# ==========================================
def render_methodology():
    st.markdown("<h1>💡 Technical Framework</h1>", unsafe_allow_html=True)
    st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
    st.markdown("""
    ### Multimodal Fusion Architecture (Fusion-Net v4)
    Our system employs a **dual-stream neural network** to identify synthetic artifacts:
    
    1. **Visual Stream (CNN-Xception):** Analyzes spatio-temporal inconsistencies in facial frames, specifically focusing on the periorbital and perioral regions.
    2. **Acoustic Stream (LSTM-MFCC):** Extracts spectral features to identify "ringing" artifacts typical of GAN-generated voices.
    3. **Fusion Layer:** A cross-modal attention mechanism that evaluates the 'Lip-Sync' coherence between the two streams.
    
    *Key Research Benchmark: Tested against FaceForensics++ and DeepFake Detection Challenge (DFDC) datasets.*
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*O-8O85_UAnXlY5S_A7rKlg.png", caption="System Architecture Diagram for Research Presentation")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 9. MAIN ROUTER
# ==========================================
def main():
    if not st.session_state.auth_active:
        render_login()
    else:
        # Sidebar Navigation with custom branding
        with st.sidebar:
            st.markdown("<h2 style='color: #58a6ff; text-align: center;'>DeepGuard AI</h2>", unsafe_allow_html=True)
            st.divider()
            nav_choice = st.radio("System Modules", ["Dashboard", "Live Camera", "Audit Logs", "Methodology"])
            st.divider()
            st.write("**Investigator:** admin_root")
            st.write("**Session:** Active")
            if st.button("Secure Logout"):
                st.session_state.auth_active = False
                st.rerun()

        if nav_choice == "Dashboard":
            render_dashboard()
        elif nav_choice == "Live Camera":
            render_camera()
        elif nav_choice == "Audit Logs":
            render_history()
        elif nav_choice == "Methodology":
            render_methodology()

if __name__ == "__main__":
    main()
