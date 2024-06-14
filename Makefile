docker_build_file := .DOCKER
docker_image_name := anterondocker/computer_vision_demos
uid := $(shell id -u)

.PHONY: run clean docker

run: $(docker_build_file)
	docker run --network host --rm -v $(abspath .):/app -it $(docker_image_name)

$(docker_build_file): Dockerfile
	docker build . -t $(docker_image_name)
	touch $(docker_build_file)

docker: $(docker_build_file)

clean:
	rm -rf $(venv)
