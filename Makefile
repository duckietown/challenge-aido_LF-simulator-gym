
build:
	dts build_utils aido-container-build --use-branch daffy --push



bump: # v2
	bumpversion patch
	git push --tags
	git push


upload:
	rm -f dist/*
	rm -rf src/*.egg-info
	python3 setup.py sdist
	twine upload --skip-existing --verbose dist/*
