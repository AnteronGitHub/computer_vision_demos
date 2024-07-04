"""Module contains various utility functions for overlaying debugging data into video frames."""
import cv2

def add_fps_counter(frame, processing_latency : float):
    """Adds and FPS counter to the upper left corner of the video frame."""
    fps = int(1/processing_latency)
    cv2.putText(frame, f"{fps} FPS", (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
