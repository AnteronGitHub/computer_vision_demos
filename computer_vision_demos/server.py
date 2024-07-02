import asyncio
import logging
import os
import time

from aiohttp import web, MultipartWriter

from .image_pipeline import ImagePipeline

class ServerStatistics:
    """Maintains counters for calculating server statistics such as average frame rate.
    """
    def __init__(self):
        self.sum_fps = 0
        self.no_frames = 0
        self.previous_output_sent = None
        self.logger = logging.getLogger("computer_vision_demos.server_statistics")

    def frame_sent(self):
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
    def __init__(self, camera_host, detect_objects : bool = True, image_pipeline_configuration = None):
        self.logger = logging.getLogger("computer_vision_demos.server")
        self.public_dir = os.path.join("computer_vision_demos", "public")

        self.image_pipeline = ImagePipeline(camera_host, detect_objects, image_pipeline_configuration)

    async def get_index(self, request):
        return web.Response(text=open(os.path.join(self.public_dir, "index.html"), 'r').read(), content_type='text/html')

    async def get_favicon(self, request):
        return web.FileResponse(os.path.join(self.public_dir, "favicon.ico"))

    async def get_stream(self, request):
        self.logger.debug("Client connected to stream")

        my_boundary = '123456789000000000000987654321'
        response = web.StreamResponse(status=200,
                                      reason='OK',
                                      headers={'Content-Type': f'multipart/x-mixed-replace;boundary={my_boundary}'})
        await response.prepare(request)
        statistics = ServerStatistics()
        try:
            while True:
                output_frame = await self.image_pipeline.frame_updated()

                with MultipartWriter('image/jpeg', boundary=my_boundary) as mpwriter:
                    mpwriter.append(output_frame, { 'Content-Type': 'image/jpeg' })
                    await mpwriter.write(response, close_boundary=False)

                statistics.frame_sent()
        except (ConnectionResetError, ConnectionError):
            pass

        self.logger.debug("Client disconnected from stream.")
        statistics.print_average_frame_rate()

        return response

    async def start_http_server(self):
        """Waits for the image pipeline to have started the stream and then starts the HTTP server.
        """
        app = web.Application()
        app.add_routes([web.get('/', self.get_index), \
                        web.get('/favicon.ico', self.get_favicon), \
                        web.get('/stream', self.get_stream)])

        await self.image_pipeline.stream_started.wait()

        self.logger.info(f"Serving on 'http://0.0.0.0:8080/'")
        await web._run_app(app, print=None)

    def start(self):
        """Starts the image pipeline and the http server instance.
        """
        loop = asyncio.get_event_loop()
        try:
            asyncio.ensure_future(self.image_pipeline.input_stream.start_input_stream())
            asyncio.ensure_future(self.image_pipeline.start_processing_pipeline())
            asyncio.ensure_future(self.start_http_server())
            loop.run_forever()
        except Exception as e:
            if repr(e) == "Event loop is closed":
                pass
        finally:
            loop.close()
