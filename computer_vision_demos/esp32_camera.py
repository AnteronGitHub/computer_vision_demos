import cv2 as cv
import logging
import requests

CONTROL_VARIABLE_FRAMESIZE = 'framesize'
CONTROL_VARIABLE_QUALITY = 'quality'

class ESP32CameraClient():
    def __init__(self, host : str):
        self.host = host
        self.cap = None
        self.logger = logging.getLogger("computer_vision_demos.esp32_camera")

    @property
    def stream_url(self):
        return f"http://{self.host}:81/stream"

    def set_control_variable(self, variable : str, value : int):
        try:
            self.logger.info(f"Setting {variable} to {value} on camera at '{self.host}'")
            requests.get(f"http://{self.host}/control?var={variable}&val={value}")
        except requests.exceptions.ConnectionError as e:
            raise e

    def connect(self):
        self.cap = cv.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise Exception(f"Could connect to camera at '{self.stream_url}'")

    def disconnect(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise "Could not read frame"

        return frame

