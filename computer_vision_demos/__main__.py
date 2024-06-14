
from dotenv import load_dotenv
import os
from .server import ComputerVisionVideoServer

if __name__ == '__main__':
    load_dotenv(dotenv_path=".env")
    camera_host = os.environ.get('ESP_CAMERA_HOST')
    detect_objects = os.environ.get('DETECT_OBJECTS') != 'no'

    if camera_host is None:
        print("No camera hosts specified. Set env var 'ESP_CAMERA_HOST' with a camera IP.")
        exit()

    ComputerVisionVideoServer(camera_host, detect_objects).start()

