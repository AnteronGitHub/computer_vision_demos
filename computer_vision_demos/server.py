from aiohttp import web, MultipartWriter
import logging
import os

from .esp32_camera import ESP32CameraClient
from .image_pipeline import ImagePipeline

class ComputerVisionVideoServer:
    def __init__(self, camera_host, detect_objects : bool = True):
        self.camera_client = ESP32CameraClient(camera_host)
        self.image_pipeline = ImagePipeline(detect_objects)
        self.public_dir = os.path.join("computer_vision_demos", "public")
        self.logger = logging.getLogger("computer_vision_demos.server")

    def start(self):
        self.camera_client.set_framesize(8)

        async def get_index(request):
            return web.Response(text=open(os.path.join(self.public_dir, "index.html"), 'r').read(), content_type='text/html')

        async def get_favicon(request):
            return web.FileResponse(os.path.join(self.public_dir, "favicon.ico"))

        async def handle(request):
            self.logger.info("Client connected to stream")
            self.camera_client.connect()

            my_boundary = '123456789000000000000987654321'
            response = web.StreamResponse(status=200,
                                          reason='OK',
                                          headers={'Content-Type': f'multipart/x-mixed-replace;boundary={my_boundary}'})
            await response.prepare(request)
            try:
                while True:
                    input_frame = self.camera_client.get_frame()
                    output_frame = self.image_pipeline.process(input_frame)

                    with MultipartWriter('image/jpeg', boundary=my_boundary) as mpwriter:
                        mpwriter.append(output_frame, { 'Content-Type': 'image/jpeg' })
                        await mpwriter.write(response, close_boundary=False)
            except ConnectionResetError:
                self.logger.info("Client disconnected from stream.")

            self.camera_client.disconnect()
            return response

        app = web.Application()
        app.add_routes([web.get('/', get_index), web.get('/favicon.ico', get_favicon), web.get('/stream', handle)])
        self.logger.info(f"Server starting on 'http://0.0.0.0:8080/'")
        web.run_app(app, print=None)

