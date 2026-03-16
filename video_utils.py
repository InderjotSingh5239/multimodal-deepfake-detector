import cv2
import numpy as np


def extract_frames(video_path, max_frames=20):

    cap = cv2.VideoCapture(video_path)

    frames = []
    count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (64, 64))
        frames.append(frame)

        count += 1
        if count >= max_frames:
            break

    cap.release()

    frames = np.array(frames)

    if len(frames) == 0:
        return np.zeros((20, 64, 64, 3))

    return frames
