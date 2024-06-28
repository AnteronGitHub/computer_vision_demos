import asyncio
import functools
import cv2
import logging
import numpy as np
import time

from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from .esp32_camera import ESP32CameraHTTPClient, ESP32CameraFrameSize, CONTROL_VARIABLE_FRAMESIZE, CONTROL_VARIABLE_QUALITY

class Debugger:
    def add_overlay(self, frame, processing_latency):
        fps = int(1/processing_latency)
        cv2.putText(frame, f"{fps} FPS", (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

class ImagePipelineConfiguration:
    def __init__(self, frame_size : ESP32CameraFrameSize = ESP32CameraFrameSize.FRAMESIZE_HD, quality : int = 10):
        self.frame_size = frame_size.value
        self.quality = quality

class ESP32CameraHTTPStream:
    def __init__(self, camera_host : str, image_pipeline_configuration = None):
        self.logger = logging.getLogger("computer_vision_demos.ESP32CameraHTTPStream")

        self.camera_client = ESP32CameraHTTPClient(camera_host)

        if image_pipeline_configuration is None:
            self.configuration = ImagePipelineConfiguration()
        else:
            self.configuration = image_pipeline_configuration

        self.executor = ThreadPoolExecutor()
        self.frame = None
        self.frame_buffered = asyncio.Condition()

    async def start_input_stream(self):
        self.logger.debug(f"Starting camera input stream")

        self.camera_client.set_control_variable(CONTROL_VARIABLE_FRAMESIZE, self.configuration.frame_size)
        self.camera_client.set_control_variable(CONTROL_VARIABLE_QUALITY, self.configuration.quality)

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

class ImagePipeline:
    """Pulls images from a source and processes them e.g. by detecting objects with a model.
    """
    def __init__(self, camera_host : str, detect_objects : bool = True, image_pipeline_configuration = None):
        self.logger = logging.getLogger("computer_vision_demos.image_pipeline")

        self.input_stream = ESP32CameraHTTPStream(camera_host, image_pipeline_configuration)

        if detect_objects:
            self.logger.info(f"Preparing YOLOv8 model for object detection.")
            self.model = YOLO()
        else:
            self.model = None

        self.stream_started = asyncio.Event()

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

    def warm_up(self):
        """Process a single frame to 'warm up' any JIT compiled kernels.
        """
        self.logger.debug("Warming up the image pipeline...")
        warm_up_started = time.time()

        input_frame = np.zeros(shape=(1200, 1600, 3), dtype=np.uint8) #.astype('uint8')
        output_frame = self.process(input_frame)

        warm_up_time = time.time() - warm_up_started
        self.logger.info(f"Warmed up the model in {warm_up_time:.2f} seconds")

    async def start_processing_pipeline(self):
        """Connects to a camera and starts pulling frames over HTTP.
        """
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(self.pipeline_executor, self.warm_up)
        self.stream_started.set()
        while True:
            input_frame = await self.input_stream.frame_updated()
            output_frame = await loop.run_in_executor(self.pipeline_executor, functools.partial(self.process, input_frame))
            async with self.output_buffered:
                self.output_frame = output_frame.copy()
                self.output_buffered.notify_all()

    async def frame_updated(self):
        """Returns an updated frame as soon as one is available. Can be awaited in e.g. loops in external async
        functions.
        """
        async with self.output_buffered:
            await self.output_buffered.wait()
            return self.output_frame.copy()

