import numpy as np


def fuse_features(video_frames, audio_features):

    video_feature = np.mean(video_frames)

    fusion_vector = np.append(video_feature, audio_features)

    return fusion_vector.reshape(1, -1)
