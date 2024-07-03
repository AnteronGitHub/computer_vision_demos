import cv2

class Debugger:
    def add_overlay(self, frame, processing_latency):
        fps = int(1/processing_latency)
        cv2.putText(frame, f"{fps} FPS", (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

from .object_detection_stream import ObjectDetectionStream
from .instance_segmentation_stream import InstanceSegmentationStream

__all__ = ["InstanceSegmentationStream", "ObjectDetectionStream"]
