YAW_THRESHOLD   = 0.3
PITCH_THRESHOLD = 0.25

def get_head_pose(lm):
    import numpy as np
    nose      = np.array([lm[1].x,   lm[1].y])
    left_eye  = np.array([lm[33].x,  lm[33].y])
    right_eye = np.array([lm[263].x, lm[263].y])
    eye_mid   = (left_eye + right_eye) / 2
    yaw       = nose[0] - eye_mid[0]
    pitch     = nose[1] - eye_mid[1]
    return yaw, pitch

def is_distracted(yaw, pitch):
    return abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD