"""Module contains various classes for using ESP32-Cam microcontrollers as data sources."""
from .esp32_camera_http import ESP32CameraHTTPStream
from .esp32_camera_frameflow import ESP32CameraFrameFlowStream

__all__ = ["ESP32CameraHTTPStream", "ESP32CameraFrameFlowStream"]
