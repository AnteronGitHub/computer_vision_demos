import asyncio
import cv2 as cv
import logging
import requests
from concurrent.futures import ThreadPoolExecutor

from .esp32_camera import ESP32CameraConfiguration, ESP32CameraControlVariable

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

class ESP32CameraHTTPStream:
    def __init__(self, camera_host : str, image_pipeline_configuration = None):
        self.logger = logging.getLogger("computer_vision_demos.ESP32CameraHTTPStream")

        self.camera_client = ESP32CameraHTTPClient(camera_host)

        if image_pipeline_configuration is None:
            self.configuration = ESP32CameraConfiguration()
        else:
            self.configuration = image_pipeline_configuration

        self.executor = ThreadPoolExecutor()
        self.frame = None
        self.frame_buffered = asyncio.Condition()

    async def start_input_stream(self):
        self.logger.debug(f"Starting camera input stream")

        self.camera_client.set_control_variable(ESP32CameraControlVariable.FRAME_SIZE, self.configuration.frame_size)
        self.camera_client.set_control_variable(ESP32CameraControlVariable.QUALITY, self.configuration.quality)

        self.camera_client.connect()

        loop = asyncio.get_running_loop()
        while True:
            input_frame = await loop.run_in_executor(self.executor, self.camera_client.get_frame)
            async with self.frame_buffered:
                self.frame = input_frame.copy()
                self.frame_buffered.notify_all()

        self.camera_client.disconnect()

    async def frame_updated(self):
        """Returns an updated frame as soon as one is available. Can be awaited in e.g. loops in external async
        functions.
        """
        async with self.frame_buffered:
            await self.frame_buffered.wait()
            return self.frame.copy()

