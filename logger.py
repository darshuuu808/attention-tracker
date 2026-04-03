import csv
import os
import time
import matplotlib.pyplot as plt

LOG_FILE = "session_log.csv"

def init_log():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ear", "yaw", "pitch", "blink_rate", "score", "status"])

def log_entry(ear, yaw, pitch, blink_rate, score, status):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round(time.time(), 2), round(ear, 3),
                         round(yaw, 3), round(pitch, 3),
                         blink_rate, score, status])

def plot_session():
    timestamps, scores, statuses = [], [], []
    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            scores.append(int(row["score"]))
            statuses.append(row["status"])

    if not timestamps:
        return

    start = timestamps[0]
    times = [t - start for t in timestamps]

    colors = []
    for s in statuses:
        if s == "ALERT":
            colors.append("green")
        elif s == "DROWSY":
            colors.append("orange")
        else:
            colors.append("red")

    plt.figure(figsize=(12, 5))
    plt.scatter(times, scores, c=colors, s=10, zorder=3)
    plt.plot(times, scores, color="gray", linewidth=0.8, alpha=0.5)
    plt.axhline(70, color="green",  linestyle="--", linewidth=0.8, label="Alert threshold")
    plt.axhline(40, color="orange", linestyle="--", linewidth=0.8, label="Drowsy threshold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Attention Score")
    plt.title("Session Attention Report")
    plt.legend()
    plt.tight_layout()
    plt.savefig("session_report.png")
    plt.show()
    print("Report saved as session_report.png")