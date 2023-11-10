#!/bin/bash

set -e

if [ -d ".venv" ]; then
    rm -rf .venv
fi

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install torch click fire questionary fairscale sentencepiece
pip install ruff mypy black