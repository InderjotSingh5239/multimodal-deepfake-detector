import streamlit as st
import tempfile

from utils.video_utils import extract_frames
from utils.audio_utils import extract_audio_features
from utils.feature_fusion import fuse_features
from utils.prediction import predict


st.title("Multimodal Deepfake Detection")

st.write("Upload a video file to analyze whether it is real or manipulated.")


uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)


if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)

    tfile.write(uploaded_file.read())

    video_path = tfile.name

    st.video(video_path)

    st.write("Processing video...")

    frames = extract_frames(video_path)

    audio_features = extract_audio_features(video_path)

    fusion_features = fuse_features(frames, audio_features)

    result = predict(fusion_features)

    st.subheader("Detection Result")

    if result == "Deepfake":
        st.error("The video appears to be a deepfake.")
    else:
        st.success("The video appears to be real.")
