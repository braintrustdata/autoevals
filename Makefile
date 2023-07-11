SHELL := /bin/bash
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
VENV_PRE_COMMIT := ${ROOT_DIR}/venv/.pre_commit

.PHONY: all
all: ${VENV_PRE_COMMIT}

.PHONY: py
py: ${VENV_PYTHON_PACKAGES}
	bash -c 'source venv/bin/activate'

VENV_INITIALIZED := venv/.initialized

${VENV_INITIALIZED}:
	rm -rf venv && python3 -m venv venv
	@touch ${VENV_INITIALIZED}

VENV_PYTHON_PACKAGES := venv/.python_packages

${VENV_PYTHON_PACKAGES}: ${VENV_INITIALIZED}
	bash -c 'source venv/bin/activate && python -m pip install --upgrade pip setuptools build twine'
	bash -c 'source venv/bin/activate && python -m pip install -e .[dev]'
	@touch $@

${VENV_PRE_COMMIT}: ${VENV_PYTHON_PACKAGES}
	bash -c 'source venv/bin/activate && pre-commit install'
	@touch $@

develop: ${VENV_PRE_COMMIT}
	@echo "--\nRun "source env.sh" to enter development mode!"

fixup:
	pre-commit run --all-files

.PHONY: test build publish clean docs

test:
	source env.sh && python3 -m pytest

build:
	source env.sh && python3 -m build

publish: build
	source env.sh && python3 -m twine upload dist/autoevals-${SDK_VERSION}*

clean:
	source env.sh && rm -rf dist/*
