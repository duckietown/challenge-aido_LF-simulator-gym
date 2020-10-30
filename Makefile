
build:
	dts build_utils aido-container-build


push: build
	dts build_utils aido-container-push


bump: # v2
	bumpversion patch
	git push --tags
	git push
