import librosa
import numpy as np


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
