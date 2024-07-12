"""This module contains implementation of push based TCP-protocol with flow control for getting video frames from a
ESP32-Cam.
"""
import asyncio
import io
import logging
import struct

import cv2
import numpy as np

from computer_vision_demos.stream import FrameStream
from .esp32_camera import ESP32CameraConfiguration

class FrameFlowReceiverProtocol(asyncio.Protocol):
    """Frame Flow receiver protocol implementatino with Asyncio.

    Frame Flow is a push based TCP-based protocol for streaming video frames from a network camera.

    Sender protocol implementation for ESP32-Cam module is available at
    https://github.com/AnteronGitHub/ESP32-CAM-FrameFlow
    """
    def __init__(self, frame_received_callback):
        self.logger = logging.getLogger("computer_vision_demos.FrameFlowProtocol")

        self.transport = None
        self.receiving_data = False
        self.data_buffer = io.BytesIO()
        self.buffer_length = 0

        self.frame_received_callback = frame_received_callback

    def _clear_buffer(self):
        self.data_buffer = io.BytesIO()
        self.receiving_data = False
        self.buffer_length = 0

    def connection_made(self, transport):
        self.transport = transport
        self.logger.info("Camera connected.")

    def data_received(self, data : bytes):
        if self.receiving_data:
            payload = data
        else:
            self.receiving_data = True
            header = data[:4]

            # ESP32-Cam uses little-endian format
            [self.buffer_length] = struct.unpack("<L", header)
            payload = data[4:]

        self.data_buffer.write(payload)

        if self.data_buffer.getbuffer().nbytes >= self.buffer_length:
            self.data_buffer.seek(0)
            payload_bytes = self.data_buffer.read(self.buffer_length)
            image = np.asarray(bytearray(payload_bytes))
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if self.data_buffer.getbuffer().nbytes - self.buffer_length == 0:
                self._clear_buffer()
            else:
                header = self.data_buffer.read(4)
                [self.buffer_length] = struct.unpack("<L", header)
                payload = self.data_buffer.read()
                self.data_buffer = io.BytesIO()
                self.data_buffer.write(payload)

            asyncio.create_task(self.frame_received_callback(frame))

            # Send server feedback
            feedback = 1024
            self.transport.write(struct.pack("@I", feedback))

    def connection_lost(self, exc):
        self.logger.info("Camera disconnected.")

class ESP32CameraFrameFlowStream(FrameStream):
    """Class implementing a video stream using frame flow over TCP.
    """
    def __init__(self, host : str = '0.0.0.0', port : int = 82):
        super().__init__()
        self.logger = logging.getLogger("computer_vision_demos.ESP32CameraFrameFlowStream")

        self.configuration = ESP32CameraConfiguration()

        self.host = host
        self.port = port

    async def frame_received_callback(self, input_frame):
        """Callback coroutine that will be called by the communication protocol when a new frame has been received.
        """
        await self.publish_frame(input_frame)

    async def start(self):
        """Starts the frame flow upstream server.
        """
        self.logger.info("Starting ESP32-Cam Frame Flow Server on '%s:%d'.", self.host, self.port)

        loop = asyncio.get_running_loop()
        server = await loop.create_server(lambda: FrameFlowReceiverProtocol(self.frame_received_callback),
                                    host=self.host,
                                    port=self.port)
        async with server:
            await server.serve_forever()
