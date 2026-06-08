SHELL := /bin/bash
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
VENV_PRE_COMMIT := ${ROOT_DIR}/.venv/.pre_commit

.PHONY: all
all: ${VENV_PRE_COMMIT}

.PHONY: py
py: ${VENV_PYTHON_PACKAGES}
	bash -c 'source .venv/bin/activate'

VENV_INITIALIZED := .venv/.initialized

${VENV_INITIALIZED}:
	rm -rf .venv && uv sync --extra dev --extra scipy
	@touch ${VENV_INITIALIZED}

VENV_PYTHON_PACKAGES := .venv/.python_packages

${VENV_PYTHON_PACKAGES}: ${VENV_INITIALIZED}
	uv sync --extra dev --extra scipy
	@touch $@

${VENV_PRE_COMMIT}: ${VENV_PYTHON_PACKAGES}
	uv run --extra dev pre-commit install
	@touch $@

develop: ${VENV_PRE_COMMIT}
	@echo "--\nRun "source env.sh" to enter development mode!"

fixup:
	pre-commit run --all-files

.PHONY: test test-py test-js

test: test-py test-js

test-py:
	uv run --extra dev --extra scipy pytest

test-js:
	pnpm install --frozen-lockfile && pnpm run test
