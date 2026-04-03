import numpy as np

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def compute_ear(landmarks, eye_idx, w, h):
    p = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C)

def average_ear(landmarks, w, h):
    left  = compute_ear(landmarks, LEFT_EYE,  w, h)
    right = compute_ear(landmarks, RIGHT_EYE, w, h)
    return (left + right) / 2.0