import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def draw_detections(img, results, class_names):
    for r in results:
        annotator = Annotator(img)
        for box in r.boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, class_names[int(c)])

    return annotator.result()

from .esp32_camera import ESP32CameraClient
from aiohttp import web, MultipartWriter

import os

class ComputerVisionVideoServer:
    def __init__(self, camera_host, detect_objects : bool = True):
        self.model = YOLO() if detect_objects else None
        self.camera_client = ESP32CameraClient(camera_host)
        self.public_dir = os.path.join("computer_vision_demos", "public")

    def start(self):
        self.camera_client.set_framesize(8)

        async def get_index(request):
            return web.Response(text=open(os.path.join(self.public_dir, "index.html"), 'r').read(), content_type='text/html')

        async def get_favicon(request):
            return web.FileResponse(os.path.join(self.public_dir, "favicon.ico"))

        async def handle(request):
            print("Client connected")
            self.camera_client.connect()

            my_boundary = '123456789000000000000987654321'
            response = web.StreamResponse(status=200,
                                          reason='OK',
                                          headers={'Content-Type': f'multipart/x-mixed-replace;boundary={my_boundary}'})
            await response.prepare(request)
            try:
                while True:
                    frame = self.camera_client.get_frame()
                    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
                    if self.model is not None:
                        detected_objects = self.model(frame, verbose=False)
                        frame = draw_detections(frame, detected_objects, self.model.names)
    
                    (flag, encodedImage) = cv.imencode(".jpg", frame)
                    data = bytearray(encodedImage)
                    with MultipartWriter('image/jpeg', boundary=my_boundary) as mpwriter:
                        mpwriter.append(data, { 'Content-Type': 'image/jpeg' })
                        await mpwriter.write(response, close_boundary=False)
            except ConnectionResetError:
                print("Connection closed by client.")

            self.camera_client.disconnect()
            return response

        app = web.Application()
        app.add_routes([web.get('/', get_index), web.get('/favicon.ico', get_favicon), web.get('/stream', handle)])
        web.run_app(app)

