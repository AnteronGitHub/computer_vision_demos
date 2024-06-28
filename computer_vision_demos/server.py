import asyncio
from aiohttp import web, MultipartWriter
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time

from .esp32_camera import ESP32CameraHTTPClient, CONTROL_VARIABLE_FRAMESIZE, CONTROL_VARIABLE_QUALITY
from .image_pipeline import ImagePipeline

class ServerStatistics:
    """Maintains counters for calculating server statistics such as average frame rate.
    """
    def __init__(self):
        self.sum_fps = 0
        self.no_frames = 0
        self.previous_output_sent = None
        self.logger = logging.getLogger("computer_vision_demos.server_statistics")

    def frame_processed(self):
        current_output_sent = time.time()
        if self.previous_output_sent is not None:
            latency = current_output_sent - self.previous_output_sent
            self.sum_fps += int(1/latency)
            self.no_frames += 1

        self.previous_output_sent = current_output_sent

    def print_average_frame_rate(self):
        if self.no_frames > 0:
            self.logger.info(f"Average frame rate: {self.sum_fps/self.no_frames:.2f} FPS")
        else:
            self.logger.info(f"No frames processed.")

class ComputerVisionVideoServer:
    """Server that pulls HTTP video frames from a ESP32 Camera HTTP Server, detects objects, and streams the output
    video to connected clients with HTTP.
    """
    def __init__(self, camera_host, detect_objects : bool = True, frame_size : int = 13, quality : int = 10):
        self.camera_client = ESP32CameraHTTPClient(camera_host)
        self.image_pipeline = ImagePipeline(detect_objects)
        self.public_dir = os.path.join("computer_vision_demos", "public")
        self.logger = logging.getLogger("computer_vision_demos.server")
        self.frame_size = frame_size
        self.quality = quality

        self.pipeline_executor = ThreadPoolExecutor()

        self.output_frame = None
        self.output_buffered = asyncio.Condition()

    def warm_up(self):
        self.logger.info("Warming up the server...")
        warm_up_started = time.time()

        self.camera_client.connect()
        input_frame = self.camera_client.get_frame()
        output_frame = self.image_pipeline.process(input_frame)
        self.camera_client.disconnect()

        warm_up_time = time.time() - warm_up_started
        self.logger.info(f"Warmed up the server in {warm_up_time:.2f} seconds")

    async def get_index(self, request):
        return web.Response(text=open(os.path.join(self.public_dir, "index.html"), 'r').read(), content_type='text/html')

    async def get_favicon(self, request):
        return web.FileResponse(os.path.join(self.public_dir, "favicon.ico"))

    async def get_stream(self, request):
        self.logger.info("Client connected to stream")

        my_boundary = '123456789000000000000987654321'
        response = web.StreamResponse(status=200,
                                      reason='OK',
                                      headers={'Content-Type': f'multipart/x-mixed-replace;boundary={my_boundary}'})
        await response.prepare(request)
        statistics = ServerStatistics()
        try:
            while True:
                async with self.output_buffered:
                    await self.output_buffered.wait()
                    output_frame = self.output_frame.copy()

                statistics.frame_processed()

                with MultipartWriter('image/jpeg', boundary=my_boundary) as mpwriter:
                    mpwriter.append(output_frame, { 'Content-Type': 'image/jpeg' })
                    await mpwriter.write(response, close_boundary=False)
        except (ConnectionResetError, ConnectionError):
            pass

        self.logger.info("Client disconnected from stream.")
        statistics.print_average_frame_rate()

        return response

    def process_frame(self):
        input_frame = self.camera_client.get_frame()
        return self.image_pipeline.process(input_frame)

    async def start_input_stream(self):
        """Connects to a camera and starts pulling frames over HTTP.
        """
        self.logger.info(f"Starting camera input stream")

        self.camera_client.set_control_variable(CONTROL_VARIABLE_FRAMESIZE, self.frame_size)
        self.camera_client.set_control_variable(CONTROL_VARIABLE_QUALITY, self.quality)

        self.camera_client.connect()

        loop = asyncio.get_running_loop()
        while True:
            output_frame = await loop.run_in_executor(self.pipeline_executor, self.process_frame)
            async with self.output_buffered:
                self.output_frame = output_frame.copy()
                self.output_buffered.notify_all()

        self.camera_client.disconnect()

    def start_http_server(self, loop):
        app = web.Application()
        app.add_routes([web.get('/', self.get_index), \
                        web.get('/favicon.ico', self.get_favicon), \
                        web.get('/stream', self.get_stream)])

        try:
            self.warm_up()
        except Exception as e:
            self.logger.error(e)
            return
        self.logger.info(f"Serving on 'http://0.0.0.0:8080/'")
        web.run_app(app, print=None, loop=loop)

    def start(self):
        loop = asyncio.get_event_loop()
        try:
            asyncio.ensure_future(self.start_input_stream())
            self.start_http_server(loop)
            loop.run_forever()
        except Exception as e:
            if repr(e) == "Event loop is closed":
                pass
        finally:
            loop.close()
