from enum import Enum

class ESP32CameraControlVariable(Enum):
    FRAME_SIZE = 'framesize'
    QUALITY = 'quality'

class ESP32CameraFrameSize(Enum):
    FRAMESIZE_96X96 = 0         # 96x96
    FRAMESIZE_QQVGA = 1         # 160x120
    FRAMESIZE_QCIF = 2          # 176x144
    FRAMESIZE_HQVGA = 3         # 240x176
    FRAMESIZE_240X240 = 4       # 240x240
    FRAMESIZE_QVGA = 5          # 320x240
    FRAMESIZE_CIF = 6           # 400x296
    FRAMESIZE_HVGA = 7          # 480x320
    FRAMESIZE_VGA = 8           # 640x480
    FRAMESIZE_SVGA = 9          # 800x600
    FRAMESIZE_XGA = 10          # 1024x768
    FRAMESIZE_HD = 11           # 1280x720
    FRAMESIZE_SXGA = 12         # 1280x1024
    FRAMESIZE_UXGA = 13         # 1600x1200

class ESP32CameraConfiguration:
    def __init__(self, frame_size : ESP32CameraFrameSize = ESP32CameraFrameSize.FRAMESIZE_HD, quality : int = 10):
        self.frame_size = frame_size.value
        self.quality = quality

