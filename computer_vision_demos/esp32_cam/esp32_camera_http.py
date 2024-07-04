"""Module containing functionality for using an ESP32-Cam with the HTTP server example application."""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import cv2
import requests

from computer_vision_demos.stream import FrameStream
from .esp32_camera import ESP32CameraConfiguration, ESP32CameraControlVariable

class ESP32CameraHTTPConnectionError(Exception):
    """Raised when the client is unable to connect to camera http server."""
    def __init__(self, url : str):
        self.url = url

    def __str__(self):
        return f"Could connect to camera at '{self.url}'"

class ESP32CameraHTTPGetFrameError(Exception):
    """Raised when the client is unable to pull a frame from a camera http server."""
    def __init__(self, url : str):
        self.url = url

    def __str__(self):
        return f"Could not read frame at '{self.url}'"

class ESP32CameraHTTPClient():
    """HTTP client for retrieving video frames from a ESP32 Camera HTTP Server.
    """
    def __init__(self, host : str):
        self.host = host
        self.cap = None
        self.logger = logging.getLogger("computer_vision_demos.esp32_cam_http")

    @property
    def stream_url(self):
        """HTTP video stream URL."""
        return f"http://{self.host}:81/stream"

    def set_control_variable(self, variable : ESP32CameraControlVariable, value : int):
        """Sets a control variable to the ESP32-Cam HTTP server."""
        try:
            self.logger.debug("Setting %d to %d on camera at '%s'", variable.value, value, self.host)
            requests.get(f"http://{self.host}/control?var={variable.value}&val={value}", timeout=10)
        except requests.exceptions.ConnectionError as e:
            raise e

    def connect(self):
        """Connects to the ESP32-Cam HTTP server."""
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise ESP32CameraHTTPConnectionError(self.stream_url)

    def disconnect(self):
        """Disconnects from the ESP32-Cam HTTP server."""
        self.cap.release()

    def get_frame(self):
        """Pulls a frame from the ESP32-Cam HTTP server.

        The used video capture buffers frames, which means that before pulling the latest frame, all of the previous
        frames need to have been pulled.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise ESP32CameraHTTPGetFrameError(self.stream_url)

        return frame

class ESP32CameraHTTPStream(FrameStream):
    """Class implementing a frame stream from video read from an ESP32-Cam HTTP server.
    """
    def __init__(self, camera_host : str):
        super().__init__()
        self.logger = logging.getLogger("computer_vision_demos.ESP32CameraHTTPStream")

        self.camera_client = ESP32CameraHTTPClient(camera_host)

        self.configuration = ESP32CameraConfiguration()

        self.executor = ThreadPoolExecutor()

    async def start(self):
        """Configures the ESP32-Cam server, and starts the video stream.
        """
        self.logger.info("Starting ESP32-Cam HTTP Stream.")

        self.camera_client.set_control_variable(ESP32CameraControlVariable.FRAME_SIZE, self.configuration.frame_size)
        self.camera_client.set_control_variable(ESP32CameraControlVariable.QUALITY, self.configuration.quality)

        self.camera_client.connect()

        loop = asyncio.get_running_loop()
        while True:
            input_frame = await loop.run_in_executor(self.executor, self.camera_client.get_frame)
            await self.publish_frame(input_frame)

        self.camera_client.disconnect()
