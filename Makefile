AIDO_REGISTRY ?= docker.io
PIP_INDEX_URL ?= https://pypi.org/simple

repo=challenge-aido_lf-simulator-gym
branch=$(shell git rev-parse --abbrev-ref HEAD)
tag=$(AIDO_REGISTRY)/duckietown/$(repo):$(branch)

update-reqs:
	pur --index-url $(PIP_INDEX_URL) -r requirements.txt -f -m '*' -o requirements.resolved
	aido-update-reqs requirements.resolved

build_options =  \
	--build-arg AIDO_REGISTRY=$(AIDO_REGISTRY) \
	--build-arg PIP_INDEX_URL=$(PIP_INDEX_URL)

build:
	docker build --pull -t $(tag) $(build_options) .

run:
	docker run -it $(tag) /bin/bash

build-no-cache:
	docker build --pull  -t $(tag) $(build_options) --no-cache .

push: build
	docker push $(tag)
