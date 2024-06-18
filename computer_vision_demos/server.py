from aiohttp import web, MultipartWriter
import logging
import os
import time

from .esp32_camera import ESP32CameraClient, CONTROL_VARIABLE_FRAMESIZE, CONTROL_VARIABLE_QUALITY
from .image_pipeline import ImagePipeline

class ComputerVisionVideoServer:
    def __init__(self, camera_host, detect_objects : bool = True, frame_size : int = 13, quality : int = 10):
        self.camera_client = ESP32CameraClient(camera_host)
        self.image_pipeline = ImagePipeline(detect_objects)
        self.public_dir = os.path.join("computer_vision_demos", "public")
        self.logger = logging.getLogger("computer_vision_demos.server")
        self.frame_size = frame_size
        self.quality = quality

    def warm_up(self):
        self.logger.info("Warming up the server...")
        warm_up_started = time.time()

        self.camera_client.connect()
        input_frame = self.camera_client.get_frame()
        output_frame = self.image_pipeline.process(input_frame)
        self.camera_client.disconnect()

        warm_up_time = time.time() - warm_up_started
        self.logger.info(f"Warmed up the server in {warm_up_time:.2f} seconds")

    def start(self):
        self.camera_client.set_control_variable(CONTROL_VARIABLE_FRAMESIZE, self.frame_size)
        self.camera_client.set_control_variable(CONTROL_VARIABLE_QUALITY, self.quality)

        async def get_index(request):
            return web.Response(text=open(os.path.join(self.public_dir, "index.html"), 'r').read(), content_type='text/html')

        async def get_favicon(request):
            return web.FileResponse(os.path.join(self.public_dir, "favicon.ico"))

        async def handle(request):
            self.logger.info("Client connected to stream")
            try:
                self.camera_client.connect()
            except Exception:
                return web.HTTPInternalServerError()

            my_boundary = '123456789000000000000987654321'
            response = web.StreamResponse(status=200,
                                          reason='OK',
                                          headers={'Content-Type': f'multipart/x-mixed-replace;boundary={my_boundary}'})
            await response.prepare(request)
            sum_fps = 0
            frames = 0
            try:
                previous_output_sent = 0
                while True:
                    input_frame = self.camera_client.get_frame()
                    output_frame = self.image_pipeline.process(input_frame)
                    current_output_sent = time.time()
                    if previous_output_sent > 0:
                        latency = current_output_sent - previous_output_sent
                        fps = int(1/latency)
                        sum_fps += fps
                        frames += 1
                    previous_output_sent = current_output_sent

                    with MultipartWriter('image/jpeg', boundary=my_boundary) as mpwriter:
                        mpwriter.append(output_frame, { 'Content-Type': 'image/jpeg' })
                        await mpwriter.write(response, close_boundary=False)
            except (ConnectionResetError, ConnectionError):
                pass

            if frames > 0:
                avg_fps = sum_fps/frames
            else:
                avg_fps = "NaN"
            self.logger.info(f"Client disconnected from stream. Average frame rate: {avg_fps:.2f} FPS")

            self.camera_client.disconnect()
            return response

        app = web.Application()
        app.add_routes([web.get('/', get_index), web.get('/favicon.ico', get_favicon), web.get('/stream', handle)])
        try:
            self.warm_up()
        except Exception as e:
            self.logger.error(e)
            return
        self.logger.info(f"Serving on 'http://0.0.0.0:8080/'")
        web.run_app(app, print=None)

