import cv2 as cv
from enum import Enum
import logging
import requests

CONTROL_VARIABLE_FRAMESIZE = 'framesize'
CONTROL_VARIABLE_QUALITY = 'quality'

class ESP32CameraFrameSize(Enum):
    FRAMESIZE_96X96 = 0         # 96x96
    FRAMESIZE_QQVGA = 1         # 160x120
    FRAMESIZE_QCIF = 2          # 176x144
    FRAMESIZE_HQVGA = 3         # 240x176
    FRAMESIZE_240X240 = 4       # 240x240
    FRAMESIZE_QVGA = 5          # 320x240
    FRAMESIZE_CIF = 6           # 400x296
    FRAMESIZE_HVGA = 7          # 480x320
    FRAMESIZE_VGA = 8           # 640x480
    FRAMESIZE_SVGA = 9          # 800x600
    FRAMESIZE_XGA = 10          # 1024x768
    FRAMESIZE_HD = 11           # 1280x720
    FRAMESIZE_SXGA = 12         # 1280x1024
    FRAMESIZE_UXGA = 13         # 1600x1200

class ESP32CameraHTTPClient():
    """HTTP client for retrieving video frames from a ESP32 Camera HTTP Server.
    """
    def __init__(self, host : str):
        self.host = host
        self.cap = None
        self.logger = logging.getLogger("computer_vision_demos.esp32_camera")

    @property
    def stream_url(self):
        return f"http://{self.host}:81/stream"

    def set_control_variable(self, variable : str, value : int):
        try:
            self.logger.debug(f"Setting {variable} to {value} on camera at '{self.host}'")
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
            raise Exception("Could not read frame")

        return frame

