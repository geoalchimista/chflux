.PHONY: init clean clean_pyc build develop docs test help

init:
	pip install -r requirements.txt

clean:
	-python setup.py clean; \

clean_pyc:
	-find . -name '*.py[co]' -exec rm {} \;
	-find . | grep -E "(__pycache__|.mypy_cache|.pytest_cache|.hypothesis|chflux.egg-info)" | xargs rm -rf \;

build: clean_pyc
	python setup.py build_ext --inplace

develop: build
	-python setup.py develop

uninstall: clean
	-python setup.py develop --uninstall

docs:
	cd docs; \
	make clean; \
	make html

test:
	pytest ./chflux/tests; \
	mypy --ignore-missing-imports ./chflux

help:
	@echo "init       initialize development environment"
	@echo "build      build the package"
	@echo "clean      clean the repository"
	@echo "clean_pyc  clean .pyc files"
	@echo "docs       generate documentation"
	@echo "test       run tests"
