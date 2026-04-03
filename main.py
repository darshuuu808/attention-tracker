import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import urllib.request
import winsound

from ear        import average_ear, LEFT_EYE, RIGHT_EYE
from head_pose  import get_head_pose, is_distracted
from scorer     import compute_score, classify
from logger     import init_log, log_entry, plot_session

# Download model
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options      = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector     = vision.FaceLandmarker.create_from_options(options)

# State
CALIBRATION_SECS = 10
calibration_ears = []
baseline_ear     = 0.0
calibrated       = False

blink_count  = 0
ear_consec   = 0
blink_times  = []
last_alert   = 0
last_log     = 0
LOG_INTERVAL = 1.0
EAR_THRESH   = 0.22
BLINK_FRAMES = 2

init_log()
cap        = cv2.VideoCapture(0)
start_time = time.time()

print("Calibrating for 10 seconds — look at the camera naturally...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    now    = time.time()
    elapsed = now - start_time

    # Dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if not calibrated:
        remaining = int(CALIBRATION_SECS - elapsed) + 1
        cv2.putText(frame, f"Calibrating... {remaining}s", (w//2 - 140, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        if result.face_landmarks:
            lm  = result.face_landmarks[0]
            avg = average_ear(lm, w, h)
            calibration_ears.append(avg)

        if elapsed >= CALIBRATION_SECS and calibration_ears:
            baseline_ear = sum(calibration_ears) / len(calibration_ears)
            calibrated   = True
            print(f"Calibration done. Baseline EAR: {baseline_ear:.3f}")

    else:
        if result.face_landmarks:
            lm = result.face_landmarks[0]

            avg_ear = average_ear(lm, w, h)

            # Blink detection
            if avg_ear < EAR_THRESH:
                ear_consec += 1
            else:
                if ear_consec >= BLINK_FRAMES:
                    blink_count += 1
                    blink_times.append(now)
                ear_consec = 0

            blink_times[:] = [t for t in blink_times if now - t < 30]
            blink_rate = len(blink_times) * 2

            yaw, pitch = get_head_pose(lm)
            score      = compute_score(avg_ear, baseline_ear, yaw, pitch, blink_rate)
            status, color = classify(score)

            # Beep alert
            if score < 40 and (now - last_alert) > 3:
                last_alert = now
                winsound.Beep(1000, 400)

            # Log every second
            if now - last_log >= LOG_INTERVAL:
                log_entry(avg_ear, yaw, pitch, blink_rate, score, status)
                last_log = now

            # Draw eye dots
            for idx in LEFT_EYE + RIGHT_EYE:
                x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            # Attention bar
            bar_w = int((score / 100) * 220)
            cv2.rectangle(frame, (20, 210), (240, 230), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 210), (20 + bar_w, 230), color, -1)
            cv2.putText(frame, f"Attention: {score}%", (20, 207),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Stats
            cv2.putText(frame, f"EAR:      {avg_ear:.3f} (base {baseline_ear:.3f})", (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(frame, f"Blinks:   {blink_count}",                           (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(frame, f"Yaw:      {yaw:.3f}",                               (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(frame, f"Pitch:    {pitch:.3f}",                             (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(frame, f"BlinkRate:{blink_rate}/min",                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            cv2.putText(frame, status, (w//2 - 90, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        else:
            cv2.putText(frame, "No Face Detected", (w//2 - 120, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Attention Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plot_session()