import asyncio
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

