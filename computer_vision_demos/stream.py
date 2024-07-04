"""Module that provides an abstraction for a continuous stream of video frames."""
import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

class FrameStream:
    """Implements a publish/subscribe stream for image frames.
    """
    def __init__(self):
        self.current_frame : np.ndarray = None
        self.frame_buffered = asyncio.Condition()

    async def publish_frame(self, new_frame : np.ndarray):
        """Returns a new tuple as soon as one is available. Can be awaited in e.g. loops in external async functions.
        """
        async with self.frame_buffered:
            self.current_frame = new_frame.copy()
            self.frame_buffered.notify_all()

    async def frame_updated(self):
        """Returns a new tuple as soon as one is available. Can be awaited in e.g. loops in external async functions.
        """
        async with self.frame_buffered:
            await self.frame_buffered.wait()
            return self.current_frame.copy()

class FrameStreamOperator:
    """An abstraction for operator that processes each tuple in an input frame stream and produces an output frame
    stream of results.
    """
    def __init__(self, input_stream : FrameStream):
        self.input_stream = input_stream
        self.started = asyncio.Event()
        self.executor = ThreadPoolExecutor()
        self.logger = logging.getLogger(f"computer_vision_demos.{self.name}")
        self.output_stream = FrameStream()

    @property
    def name(self):
        """A property representing the operator name.
        """
        return self.__class__.__name__

    def _warm_up(self):
        """Process a single frame to 'warm up' any JIT compiled kernels.
        """
        self.logger.debug("Warming up the operator...")
        warm_up_started = time.time()

        input_frame = np.zeros(shape=(1200, 1600, 3), dtype=np.uint8) #.astype('uint8')
        self.process(input_frame)

        warm_up_time = time.time() - warm_up_started
        self.logger.info("Warmed up the operator in %.2f seconds", warm_up_time)

    def process(self, frame : np.ndarray) -> np.ndarray:
        """The processing function is called for each frame from the input stream to produce frames in the output
        stream.
        """
        return frame

    async def start(self):
        """Initialies the operator runtime, and then starts the processing loop.
        """
        self.logger.info("Starting operator '%s'.", self.name)
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(self.executor, self._warm_up)
        self.started.set()
        while True:
            input_frame = await self.input_stream.frame_updated()
            output_frame = await loop.run_in_executor(self.executor, functools.partial(self.process, input_frame))
            await self.output_stream.publish_frame(output_frame)
