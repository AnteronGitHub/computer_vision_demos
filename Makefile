# User
uid := $(shell id -u)

# Python
py_requirements := requirements.txt

# Docker
docker_build_file := .DOCKER
docker_image_name := anterondocker/computer_vision_demos
ifneq (,$(shell uname -a | grep tegra))
	docker_base_image = nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
else
	docker_base_image = pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
endif

.PHONY: run run-pylint clean docker

run: $(docker_build_file)
	docker run --network host --rm -v $(abspath .):/app -it $(docker_image_name)

run-pylint: $(docker_build_file)
	docker run --network host --rm -v $(abspath .):/app -it $(docker_image_name) pylint computer_vision_demos

$(docker_build_file): Dockerfile $(py_requirements)
	docker build . --build-arg BASE_IMAGE=$(docker_base_image) -t $(docker_image_name)
	touch $(docker_build_file)

docker: $(docker_build_file)

clean:
	rm -rf $(venv)
