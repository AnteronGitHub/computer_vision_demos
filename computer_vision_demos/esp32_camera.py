import cv2 as cv
import logging
import requests

class ESP32CameraClient():
    def __init__(self, host : str):
        self.host = host
        self.cap = None
        self.logger = logging.getLogger("computer_vision_demos.esp32_camera")

    @property
    def control_url(self):
        return f"http://{self.host}/control"

    @property
    def stream_url(self):
        return f"http://{self.host}:81/stream"

    def set_framesize(self, framesize : int):
        try:
            self.logger.info(f"Setting framesize to {framesize} on camera at '{self.host}'")
            requests.get(f"{self.control_url}?var=framesize&val={framesize}")
        except requests.exceptions.ConnectionError as e:
            raise e

    def connect(self):
        self.cap = cv.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise "Could not open the capture"

    def disconnect(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise "Could not read frame"

        return frame

