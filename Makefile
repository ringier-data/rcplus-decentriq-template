PYTHON=python
PIP=pip
PYTHON_PROJECT_ROOT=.
CUDA_VISIBLE_DEVICES="-1"
VERSION=0.0.1-dev0

.PHONY: lint
lint:
	${PYTHON} -m pip install -U flake8
	${PYTHON} -m flake8 examples decentriq_deployment

.PHONY: install
install:
	PYTHONPATH=${PYTHON_PROJECT_ROOT} ${PIP} install .