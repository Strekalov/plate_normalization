.PHONY: *
DOCKER_TAG := latest
DATASET_FOLDER := dataset
DOCKER_IMAGE := hackathon_train_env
CONTAINER_NAME := hackathon_train_env
PWD = $(shell pwd)
include .env


.PHONY:
download_dataset:
	python roboflow_downloader.py				

.PHONY:
preprocess_dataset:
	python src/preprocess_dataset.py

.PHONY:
train_yolo_detector:
	python $(PWD)/src/train_letters_detector.py

.PHONY:
train_yolo_keypoints:
	python $(PWD)/src/train_letters_detector.py

.PHONY:
docker_build:
	docker build -f ./docker/Dockerfile . -t $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY:
docker_run:
	@echo "Starting docker container"
	docker run -it --rm \
		--ipc=host \
		--network=host \
		--gpus=all \
		-v $(PWD):/workspace/project \
		--name "$(CONTAINER_NAME)" \
		"$(DOCKER_IMAGE)"
