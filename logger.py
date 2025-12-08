# logger.py
import csv
import time
import os

class DataLogger:
    def __init__(self, filename="session_log.csv"):
        self.filename = filename
        self.file_exists = os.path.exists(self.filename)

        self.file = open(self.filename, "a", newline="")
        self.writer = csv.writer(self.file)

        # Scriem header doar dacă fișierul e nou
        if not self.file_exists:
            self.writer.writerow([
                "timestamp",
                "fps",
                "frame_time_ms",
                "hailo_score",
                "hailo_valid",
                "mediapipe_landmarks",
                "command",
                "tcp_sent",
                "tcp_reconnected",
                "cx",
                "cy"
            ])

    def log(self, fps, frame_time_ms, hailo_score, hailo_valid,
            mediapipe_points, command, tcp_sent, tcp_reconnected, cx, cy):

        self.writer.writerow([
            time.time(),
            fps,
            frame_time_ms,
            hailo_score,
            hailo_valid,
            mediapipe_points,
            command,
            tcp_sent,
            tcp_reconnected,
            cx,
            cy
        ])

    def close(self):
        self.file.close()
