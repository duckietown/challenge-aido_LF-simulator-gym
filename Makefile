
build:
	dts build_utils aido-container-build


push: build
	dts build_utils aido-container-push


bump: # v2
	bumpversion patch
	git push --tags
	git push


upload: # v3
	dts build_utils check-not-dirty
	dts build_utils check-tagged
	dt-check-need-upload --package duckietown-exp-manager-daffy make upload-do

upload-do:
	rm -f dist/*
	rm -rf src/*.egg-info
	python3 setup.py sdist
	twine upload --skip-existing --verbose dist/*
