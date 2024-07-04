"""Module for various computer vision application implementations."""
from .object_detection_stream import ObjectDetectionOperator
from .instance_segmentation_stream import InstanceSegmentationOperator

__all__ = ["InstanceSegmentationOperator", "ObjectDetectionOperator"]
