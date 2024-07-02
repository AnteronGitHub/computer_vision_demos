import asyncio
import cv2
import functools
import logging
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

from ..stream import FrameStream

class InstanceSegmentationStream(FrameStream):
    def __init__(self, input_stream : FrameStream):
        super().__init__()
        self.input_stream = input_stream
        self.logger = logging.getLogger("computer_vision_demos.instance_segmentation_stream")
        self.model = YOLO("yolov8n-seg.pt")
        self.executor = ThreadPoolExecutor()
        self.stream_started = asyncio.Event()

    def warm_up(self):
        """Process a single frame to 'warm up' any JIT compiled kernels.
        """
        self.logger.debug("Warming up the image pipeline...")
        warm_up_started = time.time()

        input_frame = np.zeros(shape=(1200, 1600, 3), dtype=np.uint8)
        output_frame = self.process(input_frame)

        warm_up_time = time.time() - warm_up_started
        self.logger.info(f"Warmed up the model in {warm_up_time:.2f} seconds")

    def process(self, frame : np.ndarray) -> np.ndarray:
        segments = self.model(frame, conf=.7, verbose=False)
        for r in segments:
            frame = r.plot()

        return frame

    async def start(self):
        self.logger.info("Starting instance segmentation video stream.")
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(self.executor, self.warm_up)
        self.stream_started.set()
        while True:
            input_frame = await self.input_stream.frame_updated()
            output_frame = await loop.run_in_executor(self.executor, functools.partial(self.process, input_frame))
            await self.publish_frame(output_frame)

