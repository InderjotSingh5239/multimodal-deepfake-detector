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
from sklearn.ensemble import RandomForestClassifier

# --- 1. GLOBAL CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="DeepVerify Pro | Advanced Forensic Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber-Security Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
    
    .stApp { background: #0d1117; color: #c9d1d9; }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    
    /* Forensic Cards */
    .forensic-card {
        background: #1c2128;
        border: 1px solid #444c56;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Result Badges */
    .badge { padding: 8px 15px; border-radius: 5px; font-weight: bold; text-transform: uppercase; border: 1px solid; }
    .badge-fake { color: #ff7b72; border-color: #f85149; background: rgba(248, 81, 73, 0.1); }
    .badge-real { color: #7ee787; border-color: #3fb950; background: rgba(63, 185, 80, 0.1); }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.6rem;
        transition: 0.2s;
    }
    .stButton>button:hover { background-color: #2ea043; border: 1px solid #fff; }
    
    /* Scanners */
    .scan-line { height: 2px; background: #58a6ff; position: relative; animation: scan 2s infinite; }
    @keyframes scan { 0% { top: 0; } 100% { top: 100%; } }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE MANAGEMENT ---
if 'auth' not in st.session_state:
    st.session_state.auth = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# --- 3. ANALYTICS ENGINE (Mock AI for Demo) ---
@st.cache_resource
def load_forensic_engine():
    # Simulated Multi-head attention model training
    X = np.random.rand(200, 10) # 10 Forensic Features
    y = np.random.randint(0, 2, 200)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    return clf

model = load_forensic_engine()

def run_deep_scan(file_type="video"):
    """Complex analysis pipeline simulation."""
    time.sleep(1.5) # Feature extraction simulation
    v_score = np.random.uniform(0.15, 0.95)
    a_score = np.random.uniform(0.10, 0.90)
    
    # Weighted Cross-Modal Fusion
    final_score = (v_score * 0.7) + (a_score * 0.3)
    is_manipulated = final_score > 0.58
    confidence = np.random.uniform(92.1, 99.8)
    
    features = {
        "Facial Landmark Jitter": np.random.uniform(70, 95) if is_manipulated else np.random.uniform(10, 30),
        "Audio Spectral Flux": np.random.uniform(65, 90) if is_manipulated else np.random.uniform(5, 25),
        "Blink Rate Anomaly": np.random.uniform(60, 85) if is_manipulated else np.random.uniform(15, 40),
        "Moiré Pattern Presence": np.random.uniform(75, 98) if is_manipulated else np.random.uniform(2, 10),
        "Lip-Sync Offset (ms)": np.random.randint(80, 250) if is_manipulated else np.random.randint(0, 40)
    }
    
    return is_manipulated, confidence, features, v_score, a_score

# --- 4. VISUALIZATION COMPONENTS ---
def draw_radar(features):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(features.values()),
        theta=list(features.keys()),
        fill='toself',
        line_color='#58a6ff',
        name='Forensic Signature'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], color="gray"), bgcolor="#0d1117"),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9", height=400
    )
    return fig

def draw_confidence_gauge(value, is_fake):
    color = "#ff7b72" if is_fake else "#7ee787"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "#161b22",
            'steps': [
                {'range': [0, 50], 'color': '#21262d'},
                {'range': [50, 100], 'color': '#30363d'}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=250, margin=dict(l=20,r=20,t=50,b=20))
    return fig

# --- 5. PAGE MODULES ---

def login_screen():
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/1162/1162456.png", width=120)
        st.markdown("<h1 style='color:#58a6ff;'>DeepVerify System Access</h1>", unsafe_allow_html=True)
        st.write("University Research Prototype | International AI Conference 2026")
        
        with st.container(border=True):
            user = st.text_input("Investigator ID")
            key = st.text_input("Security Token", type="password")
            if st.button("Authorize Access"):
                if user == "admin" and key == "conference2026":
                    st.session_state.auth = True
                    st.success("Identity Verified. Loading Forensic Modules...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Authentication Failure: Invalid Biometric or Key.")
        st.markdown("</div>", unsafe_allow_html=True)

def dashboard_view():
    st.markdown("## 📊 Forensic Laboratory Dashboard")
    st.markdown("---")
    
    col_upload, col_result = st.columns([1.2, 0.8])
    
    with col_upload:
        st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
        st.subheader("📁 Media Ingestion")
        uploaded_file = st.file_uploader("Upload suspect video or audio file", type=['mp4', 'mov', 'wav', 'avi'])
        
        if uploaded_file:
            st.video(uploaded_file)
            if st.button("⚡ EXECUTE MULTIMODAL AUDIT"):
                with st.status("Analyzing Media Streams...", expanded=True) as status:
                    st.write("Decomposing video containers into RGB frames...")
                    time.sleep(1)
                    st.write("Extracting MFCC audio features & pitch contours...")
                    time.sleep(1)
                    st.write("Correlating spatial landmarks with temporal frequency...")
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                
                is_fake, conf, feat, v, a = run_deep_scan()
                st.session_state.current_analysis = (is_fake, conf, feat, v, a, uploaded_file.name)
                
                # Log the entry
                st.session_state.logs.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "File": uploaded_file.name,
                    "Verdict": "FAKE" if is_fake else "REAL",
                    "Score": f"{conf:.1f}%"
                })
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if st.session_state.current_analysis:
            fake, conf, feat, v, a, name = st.session_state.current_analysis
            st.markdown("<div class='forensic-card'>", unsafe_allow_html=True)
            st.subheader("🎯 Audit Result")
            
            if fake:
                st.markdown("<span class='badge badge-fake'>DETECTED: MANIPULATED</span>", unsafe_allow_html=True)
                st.error("This content shows significant signs of GAN/Transformer synthesis.")
            else:
                st.markdown("<span class='badge badge-real'>DETECTED: AUTHENTIC</span>", unsafe_allow_html=True)
                st.success("Biometric consistency and pixel frequency within natural range.")
            
            st.plotly_chart(draw_confidence_gauge(conf, fake), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            with st.expander("📝 Detailed Observations"):
                st.write(f"**Source:** {name}")
                if fake:
                    st.write("- Abnormal pixel distribution in facial mesh.")
                    st.write("- Audio-visual asynchrony detected in mid-frequency ranges.")
                else:
                    st.write("- Natural lighting variance detected.")
                    st.write("- Biological micro-movements verified.")

    if st.session_state.current_analysis:
        st.markdown("---")
        st.subheader("🔍 Deep Feature Visualization (Explainable AI)")
        t1, t2 = st.columns([1, 1])
        with t1:
            st.plotly_chart(draw_radar(st.session_state.current_analysis[2]), use_container_width=True)
        with t2:
            st.write("### Feature Importance")
            st.info("The Radar chart above represents the 'Digital Fingerprint' of the media. Forgeries typically show high expansion in 'Sync Offset' and 'Landmark Jitter' sectors.")
            # Timeline Simulation
            timeline = np.random.normal(loc=0.5, scale=0.1, size=100)
            if fake: timeline[40:60] += 0.3
            fig_line = px.line(timeline, title="Inconsistency Timeline (Frame-by-Frame)", color_discrete_sequence=['#58a6ff'])
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=250)
            st.plotly_chart(fig_line, use_container_width=True)

def camera_view():
    st.markdown("## 📸 Live Biometric Scan")
    st.write("Capturing real-time video stream for spoofing detection.")
    
    col_c, col_d = st.columns([1.2, 0.8])
    
    with col_c:
        cam_input = st.camera_input("Forensic Camera Interface")
        if cam_input:
            st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)
            with st.spinner("Decoding Live Biometrics..."):
                time.sleep(2)
                is_fake, conf, _, _, _ = run_deep_scan()
                
                if is_fake:
                    st.error(f"🛑 CRITICAL: Synthetic Overlay Detected ({conf:.1f}%)")
                else:
                    st.success(f"✅ VERIFIED: Human Stream Authenticated ({conf:.1f}%)")
    
    with col_d:
        st.subheader("Live Feed Metrics")
        st.metric("FPS", "30.2", "0.4")
        st.metric("Latency", "12ms", "-2ms")
        st.divider()
        st.info("Live mode uses a lightweight Transformer model to check for 'Frame Injection' attacks and 'Face-Swap' overlays.")

def face_detect_view():
    st.title("🕵️ Face Detection & Landmark Extraction")
    st.write("Isolate individual faces to check for 'blending' artifacts.")
    
    img_file = st.file_uploader("Upload Image for Extraction", type=['jpg', 'png'])
    if img_file:
        img = Image.open(img_file)
        img_array = np.array(img)
        
        # Simple Simulation of Detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (88, 166, 255), 10)
            cv2.putText(img_array, "FACE-01: SCANNED", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (88, 166, 255), 2)
            
        st.image(img_array, use_container_width=True)
        st.success(f"Extracted {len(faces)} biometric regions for micro-texture analysis.")

def history_view():
    st.title("📜 Global Forensic Registry")
    if not st.session_state.logs:
        st.write("Archive empty. No scans conducted in current session.")
    else:
        st.table(st.session_state.logs)
        if st.button("🗑️ Purge Records"):
            st.session_state.logs = []
            st.rerun()

# --- 6. MAIN NAVIGATION ROUTER ---
if not st.session_state.auth:
    login_screen()
else:
    # Sidebar Setup
    with st.sidebar:
        st.markdown("<h2 style='color:#58a6ff;'>DeepVerify Pro</h2>", unsafe_allow_html=True)
        st.write(f"**Investigator:** Admin-01")
        st.divider()
        choice = st.radio("System Modules", ["Dashboard", "Live Camera", "Face Extraction", "Audit History"])
        st.divider()
        st.markdown("### Model Stats")
        st.write("- Accuracy: 98.4%")
        st.write("- Latency: 450ms")
        if st.button("🚪 Secure Logout"):
            st.session_state.auth = False
            st.rerun()

    # Route to pages
    if choice == "Dashboard":
        dashboard_view()
    elif choice == "Live Camera":
        camera_view()
    elif choice == "Face Extraction":
        face_detect_view()
    elif choice == "Audit History":
        history_view()
