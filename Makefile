AIDO_REGISTRY ?= docker.io
PIP_INDEX_URL ?= https://pypi.org/simple

repo0=$(shell basename -s .git `git config --get remote.origin.url`)
repo=$(shell echo $(repo0) | tr A-Z a-z)
branch=$(shell git rev-parse --abbrev-ref HEAD)
tag=$(AIDO_REGISTRY)/duckietown/$(repo):$(branch)

update-reqs:
	pur --index-url $(PIP_INDEX_URL) -r requirements.txt -f -m '*' -o requirements.resolved
	aido-update-reqs requirements.resolved

build_options =  \
	--build-arg AIDO_REGISTRY=$(AIDO_REGISTRY) \
	--build-arg PIP_INDEX_URL=$(PIP_INDEX_URL) \
	$(shell aido-labels)

build: update-reqs
	docker build --pull -t $(tag) $(build_options) .

build-no-cache:
	docker build --pull  -t $(tag) $(build_options) --no-cache .

push: build
	docker push $(tag)
