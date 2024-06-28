import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from .esp32_camera import ESP32CameraHTTPClient, CONTROL_VARIABLE_FRAMESIZE, CONTROL_VARIABLE_QUALITY

class Debugger:
    def add_overlay(self, frame, processing_latency):
        fps = int(1/processing_latency)
        cv2.putText(frame, f"{fps} FPS", (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

class ImagePipeline:
    """Pulls images from a source and processes them e.g. by detecting objects with a model.
    """
    def __init__(self, camera_host : str, detect_objects : bool = True, frame_size : int = 13, quality : int = 10):
        self.logger = logging.getLogger("computer_vision_demos.image_pipeline")

        self.camera_client = ESP32CameraHTTPClient(camera_host)
        self.frame_size = frame_size
        self.quality = quality

        if detect_objects:
            self.logger.info(f"Preparing YOLOv8 model for object detection.")
            self.model = YOLO()
        else:
            self.model = None

        self.debugger = Debugger()

        self.pipeline_executor = ThreadPoolExecutor()
        self.output_frame = None
        self.output_buffered = asyncio.Condition()

    def draw_detections(self, img, results):
        for r in results:
            annotator = Annotator(img)
            for box in r.boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])

        return annotator.result()

    def encode_image(self, frame) -> bytes:
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        return bytearray(encodedImage)

    def process(self, frame) -> bytes:
        processing_started = time.time()
        if self.model is not None:
            # Detect objects
            detected_objects = self.model(frame, verbose=False)

            # Draw objects
            frame = self.draw_detections(frame, detected_objects)
        e2e_latency = time.time() - processing_started

        # Add debug overlay
        self.debugger.add_overlay(frame, e2e_latency)

        # Encode the result
        encoded_image = self.encode_image(frame)
        return encoded_image

    def pull_and_process_frame(self):
        input_frame = self.camera_client.get_frame()
        return self.process(input_frame)

    def warm_up(self):
        self.logger.info("Warming up the image pipeline...")
        warm_up_started = time.time()

        self.camera_client.connect()
        input_frame = self.camera_client.get_frame()
        output_frame = self.process(input_frame)
        self.camera_client.disconnect()

        warm_up_time = time.time() - warm_up_started
        self.logger.info(f"Warmed up the server in {warm_up_time:.2f} seconds")

    async def start(self):
        """Connects to a camera and starts pulling frames over HTTP.
        """
        try:
            self.warm_up()
        except Exception as e:
            self.logger.error(e)
            return

        self.logger.info(f"Starting camera input stream")

        self.camera_client.set_control_variable(CONTROL_VARIABLE_FRAMESIZE, self.frame_size)
        self.camera_client.set_control_variable(CONTROL_VARIABLE_QUALITY, self.quality)

        self.camera_client.connect()

        loop = asyncio.get_running_loop()
        while True:
            output_frame = await loop.run_in_executor(self.pipeline_executor, self.pull_and_process_frame)
            async with self.output_buffered:
                self.output_frame = output_frame.copy()
                self.output_buffered.notify_all()

        self.camera_client.disconnect()

    async def frame_updated(self):
        async with self.output_buffered:
            await self.output_buffered.wait()
            return self.output_frame.copy()

