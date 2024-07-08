"""Module provides a front end HTTP server that can be used to access the video analysis output stream."""
import asyncio
import logging
import os
import time

import cv2
import numpy as np
from aiohttp import web, MultipartWriter

#from .esp32_cam import ESP32CameraHTTPStream
from .esp32_cam import ESP32CameraFrameFlowStream
from .video_analysis import InstanceSegmentationOperator
#from .video_analysis import ObjectDetectionOperator

class StreamStatistics:
    """Maintains counters for calculating stream statistics such as average frame rate.
    """
    def __init__(self):
        self.sum_fps = 0
        self.no_frames = 0
        self.previous_output_sent = None
        self.logger = logging.getLogger("computer_vision_demos.server_statistics")

    def frame_sent(self):
        """Called after the server has sent a frame to a connected client.
        """
        current_output_sent = time.time()
        if self.previous_output_sent is not None:
            latency = current_output_sent - self.previous_output_sent
            self.sum_fps += int(1/latency)
            self.no_frames += 1

        self.previous_output_sent = current_output_sent

    def print_average_frame_rate(self):
        """Prints the average rate for the calculated frames.
        """
        if self.no_frames > 0:
            self.logger.info("Average frame rate: %.2f FPS", self.sum_fps/self.no_frames)
        else:
            self.logger.info("No frames processed.")

class ComputerVisionVideoServer:
    """Server that pulls HTTP video frames from a ESP32 Camera HTTP Server, detects objects, and streams the output
    video to connected clients with HTTP.
    """
    def __init__(self, camera_host : str = None):
        self.logger = logging.getLogger("computer_vision_demos.server")
        self.public_dir = os.path.join("computer_vision_demos", "public")

        #self.input_stream = ESP32CameraHTTPStream(camera_host)
        self.input_stream = ESP32CameraFrameFlowStream()

        # Pick the demo application below
        #self.video_analysis_operator = ObjectDetectionOperator(self.input_stream)
        self.video_analysis_operator = InstanceSegmentationOperator(self.input_stream)

    def encode_frame(self, frame : np.ndarray) -> bytes:
        """Encodes a video frame to JPG.
        """
        (_, encoded_image) = cv2.imencode(".jpg", frame)
        return bytearray(encoded_image)

    async def get_index(
                self,
                request # pylint: disable=unused-argument
            ):
        """GET handler for the index html page containing the video viewer.
        """
        return web.Response(text=open(os.path.join(self.public_dir, "index.html"), 'r', encoding='utf-8').read(), \
                            content_type='text/html')

    async def get_favicon(
                self,
                request # pylint: disable=unused-argument
            ):
        """GET handler for the favicon.ico (requested by default in mainstream browsers).
        """
        return web.FileResponse(os.path.join(self.public_dir, "favicon.ico"))

    async def get_stream(
                self,
                request # pylint: disable=unused-argument
            ):
        """GET handler for video stream.

        Starts an HTTP video stream running until the client disconnects.
        """
        self.logger.debug("Client connected to stream")

        my_boundary = '123456789000000000000987654321'
        response = web.StreamResponse(status=200,
                                      reason='OK',
                                      headers={'Content-Type': f'multipart/x-mixed-replace;boundary={my_boundary}'})
        await response.prepare(request)
        statistics = StreamStatistics()
        try:
            while True:
                output_frame = await self.video_analysis_operator.output_stream.frame_updated()
                encoded_frame = self.encode_frame(output_frame)

                with MultipartWriter('image/jpeg', boundary=my_boundary) as mpwriter:
                    mpwriter.append(encoded_frame, { 'Content-Type': 'image/jpeg' })
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

        await self.video_analysis_operator.started.wait()

        self.logger.info("Serving on 'http://0.0.0.0:8080/'")
        await web._run_app(app, print=None)

    def start(self):
        """Starts the image pipeline and the http server instance.
        """
        loop = asyncio.get_event_loop()
        try:
            asyncio.ensure_future(self.input_stream.start())
            asyncio.ensure_future(self.video_analysis_operator.start())
            asyncio.ensure_future(self.start_http_server())
            loop.run_forever()
        except RuntimeError as e:
            if repr(e) == "Event loop is closed":
                pass
        finally:
            loop.close()
