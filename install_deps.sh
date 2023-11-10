#!/bin/bash

if [ -d ".venv" ]; then
    rm -rf .venv
fi

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e ./llama
pip install -e ./llama_steve
pip install ruff mypy black