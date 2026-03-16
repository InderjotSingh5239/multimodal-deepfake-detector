Multimodal Deepfake Detection Using Audio-Visual Feature Fusion

This project demonstrates a prototype deepfake detection system that
combines visual and audio features from a video.

Pipeline

1. Upload video
2. Extract frames using OpenCV
3. Extract audio features using Librosa
4. Fuse features
5. Predict using a neural network
6. Display result with Streamlit

Run the project

1. Install dependencies

pip install -r requirements.txt

2. Train model

python train_model.py

3. Start application

streamlit run app.py
