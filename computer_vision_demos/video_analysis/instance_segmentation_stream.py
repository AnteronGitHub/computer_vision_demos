"""Module contains instance segmentation stream operator implementation."""
import time

import numpy as np
from ultralytics import YOLO

from computer_vision_demos.stream import FrameStreamOperator
from .debugger import add_fps_counter

class InstanceSegmentationOperator(FrameStreamOperator):
    """Frame stream operator that segments instances detected from frames and overlays frames with detected segments
    bounding boxes, their classes and prediction confidence. The segments are also highlighted with color."""
    def __init__(self, *args, prediction_confidence : float = .25):
        super().__init__(*args)
        self.prediction_confidence = prediction_confidence

        self.model = YOLO("yolov8n-seg.pt")

    def process(self, frame : np.ndarray) -> np.ndarray:
        processing_started = time.time()
        segments = self.model(frame, conf=self.prediction_confidence, verbose=False)
        for r in segments:
            frame = r.plot()
        e2e_latency = time.time() - processing_started

        add_fps_counter(frame, e2e_latency)

        return frame
