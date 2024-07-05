ARG BASE_IMAGE=pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM $BASE_IMAGE

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install ultralytics --no-deps

COPY requirements_yolo.txt requirements_yolo.txt
RUN pip3 install -r requirements_yolo.txt

COPY computer_vision_demos computer_vision_demos

ENV PYTHONPATH=/app

# RUN pip3 install supervision --no-deps
# RUN pip3 install defusedxml
# RUN pip3 install scipy

CMD ["python3", "-m", "computer_vision_demos"]
