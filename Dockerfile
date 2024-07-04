FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install ultralytics --no-deps

COPY requirements_yolo.txt requirements_yolo.txt
RUN pip3 install -r requirements_yolo.txt

COPY computer_vision_demos computer_vision_demos

ENV PYTHONPATH=$PYTHONPATH:/app

# RUN pip3 install supervision --no-deps
# RUN pip3 install defusedxml
# RUN pip3 install scipy

CMD ["python3", "-m", "computer_vision_demos"]
