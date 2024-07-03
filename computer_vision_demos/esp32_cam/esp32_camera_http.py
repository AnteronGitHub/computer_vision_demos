import asyncio
import cv2
import logging
import requests
from concurrent.futures import ThreadPoolExecutor

from .esp32_camera import ESP32CameraConfiguration, ESP32CameraControlVariable
from ..stream import FrameStream

class ESP32CameraHTTPClient():
    """HTTP client for retrieving video frames from a ESP32 Camera HTTP Server.
    """
    def __init__(self, host : str):
        self.host = host
        self.cap = None
        self.logger = logging.getLogger("computer_vision_demos.esp32_cam_http")

    @property
    def stream_url(self):
        return f"http://{self.host}:81/stream"

    def set_control_variable(self, variable : ESP32CameraControlVariable, value : int):
        try:
            self.logger.debug(f"Setting {variable.value} to {value} on camera at '{self.host}'")
            requests.get(f"http://{self.host}/control?var={variable.value}&val={value}")
        except requests.exceptions.ConnectionError as e:
            raise e

    def connect(self):
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise Exception(f"Could connect to camera at '{self.stream_url}'")

    def disconnect(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Could not read frame")

        return frame

class ESP32CameraHTTPStream(FrameStream):
    def __init__(self, camera_host : str):
        super().__init__()
        self.logger = logging.getLogger("computer_vision_demos.ESP32CameraHTTPStream")

        self.camera_client = ESP32CameraHTTPClient(camera_host)

        self.configuration = ESP32CameraConfiguration()

        self.executor = ThreadPoolExecutor()

    async def start(self):
        self.logger.info(f"Starting ESP32-Cam HTTP Stream.")

        self.camera_client.set_control_variable(ESP32CameraControlVariable.FRAME_SIZE, self.configuration.frame_size)
        self.camera_client.set_control_variable(ESP32CameraControlVariable.QUALITY, self.configuration.quality)

        self.camera_client.connect()

        loop = asyncio.get_running_loop()
        while True:
            input_frame = await loop.run_in_executor(self.executor, self.camera_client.get_frame)
            await self.publish_frame(input_frame)

        self.camera_client.disconnect()

