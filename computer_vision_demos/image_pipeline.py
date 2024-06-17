import cv2 as cv
import logging
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class ImagePipeline:
    def __init__(self, detect_objects : bool = True):
        self.logger = logging.getLogger("computer_vision_demos.image_pipeline")
        if detect_objects:
            self.logger.info(f"Preparing YOLOv8 model for object detection.")
            self.model = YOLO()
        else:
            self.model = None

    def draw_detections(self, img, results):
        for r in results:
            annotator = Annotator(img)
            for box in r.boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])

        return annotator.result()

    def process(self, frame) -> bytes:
        # Detect objects
        if self.model is not None:
            detected_objects = self.model(frame, verbose=False)
            frame = self.draw_detections(frame, detected_objects)

        # Encode result frame
        (flag, encodedImage) = cv.imencode(".jpg", frame)
        return bytearray(encodedImage)

