SHELL := /bin/bash
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
VENV_PRE_COMMIT := ${ROOT_DIR}/venv/.pre_commit

.PHONY: all mise-install
all: ${VENV_PRE_COMMIT}

.PHONY: mise-install
mise-install:
	@command -v mise >/dev/null 2>&1 || { echo "Error: mise is not installed. Visit https://mise.jdx.dev/getting-started.html"; exit 1; }
	mise install

.PHONY: py
py: ${VENV_PYTHON_PACKAGES}
	bash -c 'source venv/bin/activate'

VENV_INITIALIZED := venv/.initialized

${VENV_INITIALIZED}:
	rm -rf venv && python -m venv venv
	@touch ${VENV_INITIALIZED}

VENV_PYTHON_PACKAGES := venv/.python_packages

${VENV_PYTHON_PACKAGES}: ${VENV_INITIALIZED}
	bash -c 'source venv/bin/activate && python -m pip install --upgrade pip setuptools build twine openai'
	bash -c 'source venv/bin/activate && python -m pip install -e ".[dev]"'
	bash -c 'source venv/bin/activate && python -m pip install -e ".[scipy]"'  # for local tests
	@touch $@

${VENV_PRE_COMMIT}: ${VENV_PYTHON_PACKAGES}
	bash -c 'source venv/bin/activate && pre-commit install'
	@touch $@

develop: mise-install ${VENV_PRE_COMMIT}
	@echo "--\nRun "source env.sh" to enter development mode!"

fixup:
	pre-commit run --all-files

.PHONY: test test-py test-js

test: test-py test-js

test-py:
	source env.sh && python3 -m pytest

test-js:
	pnpm install && pnpm run test
