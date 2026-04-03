YAW_THRESHOLD   = 0.3
PITCH_THRESHOLD = 0.25

def compute_score(avg_ear, baseline_ear, yaw, pitch, blink_rate):
    score = 100

    ear_ratio = avg_ear / baseline_ear if baseline_ear > 0 else 1.0
    if ear_ratio < 0.75:
        score -= 40
    elif ear_ratio < 0.88:
        score -= 15

    if abs(yaw) > YAW_THRESHOLD:
        score -= 30
    if abs(pitch) > PITCH_THRESHOLD:
        score -= 20

    if blink_rate > 25:
        score -= 20
    elif blink_rate > 15:
        score -= 10

    return max(0, min(100, score))

def classify(score):
    if score >= 70:
        return "ALERT",      (0, 255, 0)
    elif score >= 40:
        return "DROWSY",     (0, 165, 255)
    else:
        return "DISTRACTED", (0, 0, 255)