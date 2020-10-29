AIDO_REGISTRY ?= docker.io
PIP_INDEX_URL ?= https://pypi.org/simple

repo0=$(shell basename -s .git `git config --get remote.origin.url`)
repo=$(shell echo $(repo0) | tr A-Z a-z)
branch=$(shell git rev-parse --abbrev-ref HEAD)
tag=$(AIDO_REGISTRY)/duckietown/$(repo):$(branch)

update-reqs:
	# dts build_utils update-reqs -r requirements.txt -o requirements.resolved
	pur --index-url $(PIP_INDEX_URL) -r requirements.txt -f -m '*' -o requirements.resolved

check-update-reqs:
	dt-update-reqs requirements.resolved

build_options =  \
	--build-arg AIDO_REGISTRY=$(AIDO_REGISTRY) \
	--build-arg PIP_INDEX_URL=$(PIP_INDEX_URL) \
	$(shell dts -q build_utils labels)


bump: # v2
	bumpversion patch
	git push --tags
	git push

build: update-reqs
	dts -q build_utils check-not-dirty
	dts -q build_utils check-tagged
	docker build --pull -t $(tag) $(build_options) .


push: build
	dts -q build_utils push $(tag)
