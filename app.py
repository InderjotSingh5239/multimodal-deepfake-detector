import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
from sklearn.linear_model import LogisticRegression

# ---------------------------------------
# Title
# ---------------------------------------

st.title("Multimodal Deepfake Detection System")

st.write(
"""
This prototype analyzes both **visual frames** and **audio signals**
from a video to determine whether it is real or manipulated.
"""
)

# ---------------------------------------
# Create a simple ML model
# (for prototype demonstration)
# ---------------------------------------

X = np.random.rand(200, 14)
y = np.random.randint(0, 2, 200)

model = LogisticRegression()
model.fit(X, y)

# ---------------------------------------
# Video Frame Feature Extraction
# ---------------------------------------

def extract_video_features(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (64,64))

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return 0

    frames = np.array(frames)

    return np.mean(frames)

# ---------------------------------------
# Audio Feature Extraction (MFCC)
# ---------------------------------------

def extract_audio_features(video_path):

    try:

        audio, sr = librosa.load(video_path)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

        features = np.mean(mfcc, axis=1)

        return features

    except:
        return np.zeros(13)

# ---------------------------------------
# Feature Fusion
# ---------------------------------------

def fuse_features(video_feature, audio_features):

    fusion_vector = np.append(video_feature, audio_features)

    return fusion_vector.reshape(1,-1)

# ---------------------------------------
# Prediction
# ---------------------------------------

def detect_deepfake(features):

    prediction = model.predict(features)

    if prediction[0] == 1:
        return "Deepfake"
    else:
        return "Real"

# ---------------------------------------
# Streamlit UI
# ---------------------------------------

uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4","avi","mov"]
)

if uploaded_file is not None:

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    temp_file.write(uploaded_file.read())

    video_path = temp_file.name

    st.video(video_path)

    st.write("Processing video...")

    # extract visual features
    video_feature = extract_video_features(video_path)

    # extract audio features
    audio_features = extract_audio_features(video_path)

    # fuse features
    fusion_features = fuse_features(video_feature, audio_features)

    # prediction
    result = detect_deepfake(fusion_features)

    st.subheader("Detection Result")

    if result == "Deepfake":
        st.error("The video appears to be a Deepfake.")
    else:
        st.success("The video appears to be Real.")
