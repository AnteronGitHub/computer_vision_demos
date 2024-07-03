from dotenv import load_dotenv
import logging
import os
from .server import ComputerVisionVideoServer

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s', level=logging.INFO)
    load_dotenv(dotenv_path=".env")
    camera_host = os.environ.get('ESP_CAMERA_HOST')

    if camera_host is None:
        print("No camera hosts specified. Set env var 'ESP_CAMERA_HOST' with a camera IP.")
        exit()

    ComputerVisionVideoServer(camera_host).start()

