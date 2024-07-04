"""Module contains object detection stream operator implementation."""
import time

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from computer_vision_demos.stream import FrameStreamOperator
from .debugger import add_fps_counter

class ObjectDetectionOperator(FrameStreamOperator):
    """Detects objects from an input stream.
    """
    def __init__(self, *args, prediction_confidence : float = .25):
        super().__init__(*args)

        self.prediction_confidence = prediction_confidence

        self.logger.debug("Preparing YOLOv8 model for object detection.")
        self.model = YOLO()

    def draw_detections(self, img, results):
        """Overlays the detected object boxes to the input image.
        """
        for r in results:
            annotator = Annotator(img)
            for box in r.boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])

        return annotator.result()

    def process(self, frame : np.ndarray) -> np.ndarray:
        processing_started = time.time()
        if self.model is not None:
            # Detect objects
            detected_objects = self.model(frame, conf=self.prediction_confidence, verbose=False)

            # Draw objects
            frame = self.draw_detections(frame, detected_objects)
        e2e_latency = time.time() - processing_started

        add_fps_counter(frame, e2e_latency)

        return frame
