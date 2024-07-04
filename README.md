# Computer Vision Demonstrations for Video
This repository includes computer vision demonstrations for video.

The demo uses ESP32-CAM module for camera and NVIDIA AGX Xavier for server-side analysis.

Object detection is implemented using YOLOv8 model.

## Preparation
To set up the demo, the analysis server needs to be in the same WLAN as the camera. The camera needs to serve the video over http (using the example http camera server code works).

Install the server dependencies by running
```
make docker
```

## Configuration
Configure the server with environment variables. An environment file `.env` can be placed at the repository root with the configuration parameters.

| Environment variable | Description                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| ESP_CAMERA_HOST      | ESP32-CAM server IP address                                               |
| DETECT_OBJECTS       | Set to 'no' to prevent object detection and simply proxy the camera video |

## Run server
After preparations, start the server by running the following command at the repository root.
```
make run
```

During the initial run, pretrained YOLOv8 model parameters will be downloaded, requiring Internet access; the parameters will be stored at the repo root, and no internet connection is needed in subsequent runs.

## Run pylint
Pylint can be run with make:
```
make run-pylint
```
