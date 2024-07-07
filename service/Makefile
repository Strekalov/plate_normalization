APP_PORT := 8888
DOCKER_TAG := latest
DOCKER_IMAGE := plate_normalization_service
NUM_WORKERS := 4

lint:
	ruff check --fix

run_app:
	python -m uvicorn app:create_app --host='0.0.0.0' --port=$(APP_PORT) --workers=$(NUM_WORKERS)

build_docker:
	docker build -f Dockerfile . --force-rm=true -t $(DOCKER_IMAGE):$(DOCKER_TAG)

run_docker_app:
	@echo "Starting service in container"
	docker run \
		--ipc=host \
		--network=host \
		--gpus=all \
		-v $(PWD)/config:/plate_normalization_service/config \
		--name "$(CONTAINER_NAME)" \
		--restart always \
		-d \
		"$(DOCKER_IMAGE):$(DOCKER_TAG)"

build_trt_engine:
	docker run \
		--ipc=host \
		--network=host \
		--gpus=all \
		-v $(PWD)/weights:/plate_normalization_service/weights \
		--name "$(CONTAINER_NAME)" \
		"$(DOCKER_IMAGE):$(DOCKER_TAG)" \
		python export_yolo_to_trt.py

locust_test:
	locust -f locustfile.py --host=http://localhost:8888 --csv=results --logfile=locust.log