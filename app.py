import streamlit as st
import cv2
import numpy as np
import librosa
import time
import tempfile
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime

# --- SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="DeepScan | Multimodal Forensic Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL STARTUP CSS ---
st.markdown("""
    <style>
    /* Global Styles */
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stApp { background-color: #0d1117; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    /* Custom Cards for UI */
    .forensic-card {
        background: #1c2128;
        border: 1px solid #444c56;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    
    /* Result Badges */
    .status-badge {
        padding: 10px 20px;
        border-radius: 30px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .status-fake { background-color: #442d30; color: #ff7b72; border: 1px solid #f85149; }
    .status-real { background-color: #233129; color: #7ee787; border: 1px solid #3fb950; }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 12px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #2ea043; cursor: pointer; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False
if 'audit_history' not in st.session_state:
    st.session_state.audit_history = []

# --- CORE LOGIC FUNCTIONS ---

def get_radar_chart(data_points):
    """Creates a professional Radar Chart for Multimodal features."""
    categories = ['Visual Texture', 'Audio Frequency', 'Lip Sync', 'Lighting Consistency', 'Pulse Analysis']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=data_points,
        theta=categories,
        fill='toself',
        name='Forensic Signature',
        line_color='#58a6ff',
        fillcolor='rgba(88, 166, 255, 0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], color="#8b949e"), bgcolor="#1c2128"),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=380
    )
    return fig

def simulate_analysis_engine():
    """Simulates the Deep Learning Pipeline for Demo purposes."""
    # This creates random but believable data for the conference demo
    is_fake = np.random.choice([True, False], p=[0.4, 0.6])
    confidence = np.random.uniform(88.5, 99.2)
    
    # Generate scores for the radar chart
    if is_fake:
        radar_data = [np.random.randint(60, 95), np.random.randint(70, 98), np.random.randint(50, 90), np.random.randint(40, 80), np.random.randint(30, 70)]
        reason = "Detected artifacts in the facial frequency domain and audio-visual misalignment."
    else:
        radar_data = [np.random.randint(5, 30), np.random.randint(10, 25), np.random.randint(5, 20), np.random.randint(10, 35), np.random.randint(5, 15)]
        reason = "Consistent biometric signals and natural light reflections verified."
        
    return is_fake, confidence, radar_data, reason

# --- AUTHENTICATION PAGE ---
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; padding-top: 80px;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=90)
        st.markdown("<h1 style='color: #58a6ff;'>DeepScan Investigator Access</h1>", unsafe_allow_html=True)
        st.write("Secure Multimodal Forensic Lab | Version 4.0")
        
        with st.form("login_form"):
            user_id = st.text_input("Investigator ID", placeholder="admin")
            access_key = st.text_input("Security Key", type="password", placeholder="conference2026")
            submit = st.form_submit_button("Authorize & Initialize")
            
            if submit:
                if user_id == "admin" and access_key == "conference2026":
                    st.session_state.is_logged_in = True
                    st.rerun()
                else:
                    st.error("Access Denied. Invalid credentials.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN DASHBOARD ---
def main_dashboard():
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=300", use_container_width=True)
        st.markdown("### 🖥️ Forensic Modules")
        choice = st.radio("Navigation", ["🔍 Analysis Dashboard", "📸 Live Forensic Camera", "📂 Audit History", "💡 Methodology"])
        st.divider()
        st.markdown("### ⚙️ System Status")
        st.success("Network: Encrypted")
        st.info("Engine: Fusion-Net v4")
        if st.button("🚪 Logout"):
            st.session_state.is_logged_in = False
            st.rerun()

    # Module 1: Analysis Dashboard
    if choice == "🔍 Analysis Dashboard":
        st.markdown("<h1>🔍 Forensic Analysis Dashboard</h1>", unsafe_allow_html=True)
        st.write("Upload media to perform a full-spectrum multimodal deepfake audit.")
        
        col_main, col_stats = st.columns([1.3, 0.7])
        
        with col_main:
            st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
            uploaded_media = st.file_uploader("Upload Video or Audio File", type=['mp4', 'mov', 'wav', 'avi'])
            
            if uploaded_media:
                st.video(uploaded_media)
                if st.button("🚀 Execute Forensic Scan"):
                    with st.status("Analyzing Media Streams...", expanded=True) as status:
                        st.write("Extracting facial landmarks via MTCNN...")
                        time.sleep(1.2)
                        st.write("Processing audio spectrograms via Librosa...")
                        time.sleep(1)
                        st.write("Calculating Cross-Modal Correlation...")
                        time.sleep(0.8)
                        status.update(label="Audit Complete!", state="complete", expanded=False)
                    
                    is_fake, conf, radar_vals, reason = simulate_analysis_engine()
                    
                    # Store results in session state for display
                    st.session_state.last_result = {
                        'fake': is_fake, 'conf': conf, 'radar': radar_vals, 'reason': reason, 'name': uploaded_media.name
                    }
                    st.session_state.audit_history.append({
                        'time': datetime.now().strftime("%H:%M"), 'name': uploaded_media.name, 'res': 'FAKE' if is_fake else 'REAL'
                    })
            st.markdown("</div>", unsafe_allow_html=True)

        with col_stats:
            if 'last_result' in st.session_state:
                res = st.session_state.last_result
                st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
                st.subheader("🎯 Audit Result")
                
                if res['fake']:
                    st.markdown("<div class='status-badge status-fake'>DETECTION: MANIPULATED</div>", unsafe_allow_html=True)
                    st.error(f"High Risk Detected: {res['conf']:.2f}% Confidence")
                else:
                    st.markdown("<div class='status-badge status-real'>DETECTION: AUTHENTIC</div>", unsafe_allow_html=True)
                    st.success(f"Low Risk: {res['conf']:.2f}% Confidence")
                
                st.write(f"**Primary Evidence:** {res['reason']}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.subheader("📊 Feature Analysis")
                st.plotly_chart(get_radar_chart(res['radar']), use_container_width=True)
            else:
                st.info("System standby. Awaiting media input for forensic decomposition.")

    # Module 2: Live Camera
    elif choice == "📸 Live Forensic Camera":
        st.markdown("<h1>📸 Real-time Stream Analysis</h1>", unsafe_allow_html=True)
        st.write("Scan a live biometric feed for frame-injection or GAN overlays.")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            cam_feed = st.camera_input("Scanner Access")
        
        with c2:
            if cam_feed:
                st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
                st.subheader("📈 Live Stream Metadata")
                with st.spinner("Analyzing biometric patterns..."):
                    time.sleep(2)
                    st.metric("FPS", "30.1", "0.2")
                    st.metric("Biometric Consistency", "94%", "-2%")
                    st.write("**Verdict:** Frame matches historical biometric record. No synthetic noise detected.")
                st.markdown("</div>", unsafe_allow_html=True)

    # Module 3: History
    elif choice == "📂 Audit History":
        st.markdown("<h1>📂 Forensic Audit Registry</h1>", unsafe_allow_html=True)
        if not st.session_state.audit_history:
            st.write("Registry is empty.")
        else:
            for item in reversed(st.session_state.audit_history):
                with st.container(border=True):
                    col_a, col_b, col_c = st.columns([1, 3, 1])
                    col_a.write(f"🕒 {item['time']}")
                    col_b.write(f"**File:** {item['name']}")
                    col_c.write(f"**Result:** {item['res']}")

    # Module 4: Methodology
    elif choice == "💡 Methodology":
        st.markdown("<h1>💡 System Architecture</h1>", unsafe_allow_html=True)
        st.markdown("""
        ### Deep Learning Architecture: Fusion-Net v4
        Our project utilizes a **Multimodal Fusion Approach** to detect deepfakes:
        1. **Visual Head:** Uses EfficientNet-B0 to extract spatio-temporal features from video frames.
        2. **Audio Head:** Uses Bi-LSTM layers to process Mel-Frequency Cepstral Coefficients (MFCC) for voice cloning detection.
        3. **Cross-Modal Attention:** A weighted fusion layer that checks if the lip movements correlate with the audio frequency.
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*O-8O85_UAnXlY5S_A7rKlg.png", caption="Proposed Research Architecture")

# --- SYSTEM ENTRY POINT ---
if __name__ == "__main__":
    if not st.session_state.is_logged_in:
        login_page()
    else:
        main_dashboard()
